"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass
from math import ceil

import torch
import torch.nn as nn
from torch.nn import functional as F

from benchmark.cuda_timer import CudaTimer

from vllm import activation_ops, layernorm_ops


class SiluAndMul(nn.Module):
    """An activation function for SwiGLU.

    The function computes x -> silu(x[:d]) * x[d:] where d = x.shape[1] // 2.

    Shapes:
        x: (num_tokens, 2 * d)
        return: (num_tokens, d)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_tokens = x.shape[0]
        d = x.shape[1] // 2
        out = torch.empty(num_tokens, d, dtype=x.dtype, device=x.device)
        activation_ops.silu_and_mul(out, x)
        return out


class RMSNorm(nn.Module):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        layernorm_ops.rms_norm(
            out,
            x,
            self.weight.data,
            self.variance_epsilon,
        )
        return out



class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_embd % config.num_tensor_parallel_workers == 0
        # key, query, value projections for all heads, but in a batch
        self._num_tensor_parallel_workers = config.num_tensor_parallel_workers
        self.n_q_head_worker = config.n_head // self._num_tensor_parallel_workers
        self.n_kv_head_worker = ceil(config.n_kv_head / self._num_tensor_parallel_workers)
        total_head_worker = self.n_q_head_worker + 2 * self.n_kv_head_worker
        self.head_dim = config.n_embd // config.n_head

        self.c_attn = nn.Linear(config.n_embd, total_head_worker * self.head_dim, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(self.n_q_head_worker * self.head_dim, config.n_embd, bias=config.bias)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
    
        self.attn_pre_proj_timer = CudaTimer("attn_pre_proj")
        self.attn_post_proj_timer = CudaTimer("attn_post_proj")

    def forward(self, x):
        with self.attn_pre_proj_timer:
            qkv = F.linear(x, self.c_attn.weight)

        num_tokens = x.shape[0]

        o = qkv.view(-1)[:num_tokens * self.n_q_head_worker * self.head_dim].view(num_tokens, -1)

        with self.attn_post_proj_timer:
            # output projection
            z = self.c_proj(o)

        return z


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.num_tensor_parallel_workers == 0

        self.use_gated_mlp = config.use_gated_mlp
        if self.use_gated_mlp:
            self.c_fc    = nn.Linear(config.n_embd, 2 * config.n_expanded_embd // config.num_tensor_parallel_workers, bias=config.bias)
            self.act = SiluAndMul()
        else:
            self.c_fc    = nn.Linear(config.n_embd, config.n_expanded_embd // config.num_tensor_parallel_workers, bias=config.bias)
            self.act = nn.GELU()

        self.c_proj  = nn.Linear(config.n_expanded_embd // config.num_tensor_parallel_workers, config.n_embd, bias=config.bias)

        self.mlp_up_proj_timer = CudaTimer("mlp_up_proj")
        self.mlp_act_timer = CudaTimer("mlp_act")
        self.mlp_down_proj_timer = CudaTimer("mlp_down_proj")
            
    def forward(self, x):
        with self.mlp_up_proj_timer:
            x = self.c_fc(x)
        
        with self.mlp_act_timer:
            x = self.act(x)

        with self.mlp_down_proj_timer:
            x = self.c_proj(x)

        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.emb = nn.Embedding(config.vocab_size // config.num_tensor_parallel_workers, config.n_embd)
        self.ln = nn.LayerNorm(config.n_embd)
        self.rms_norm = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.deemb = nn.Linear(config.n_embd, config.vocab_size // config.num_tensor_parallel_workers, bias=config.bias)

        self.emb_timer = CudaTimer("emb")
        self.layer_norm_timer = CudaTimer("layer_norm")
        self.rms_norm_timer = CudaTimer("rms_norm")
        self.deemb_timer = CudaTimer("deemb")        
        self.add_norm_timer = CudaTimer("add_norm")
            
    def forward(self, x):
        with self.emb_timer:
            x = self.emb(x)

        # x = self.attn(x)
        # x1 = self.mlp(x)

        x1 = x.clone()

        with self.add_norm_timer:
            x = x + x1

        # with self.layer_norm_timer:
        #     x = self.ln(x)

        # with self.rms_norm_timer:
        #     x = self.rms_norm(x)

        # with self.deemb_timer:
        #     x = self.deemb(x)
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 12
    n_embd: int = 768
    n_expanded_embd: int = 3072
    use_gated_mlp: bool = False
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    num_tensor_parallel_workers: int = 1
