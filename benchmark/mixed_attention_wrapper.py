import traceback
from math import ceil

import torch

#from benchmark.cuda_timer import CudaTimer
from benchmark.timer_stats_store import TimerStatsStore
from benchmark.vllm_attention import PagedAttentionWithRoPE, InputMetadata

WARMUP_STEPS = 5
ACTIVE_STEPS = 10


def _pad_to_alignment(x, multiple_of):
    return x + [0] * ((-len(x)) % multiple_of)

def _pad_length(x, multiple_of):
    return x + (-x % multiple_of)


class MixedAttentionWrapper:
    def __init__(
        self,
        batch_size,
        n_q_head,
        n_kv_head,
        n_embd,
        block_size,
        num_tensor_parallel_workers,
        prefill_chunk_size,
        kv_cache_size,
        max_context_len,
        is_prefill
    ):
        self._n_embd = n_embd
        self._n_q_head = n_q_head
        self._n_kv_head = n_kv_head
        self._block_size = block_size
        self._num_tensor_parallel_workers = num_tensor_parallel_workers
        self._head_dim = n_embd // n_q_head
        self._scaling = self._head_dim**-0.5
        assert n_q_head % num_tensor_parallel_workers == 0
        self._n_worker_q_heads = n_q_head // num_tensor_parallel_workers
        assert n_kv_head % num_tensor_parallel_workers == 0
        self._n_worker_kv_heads = n_kv_head // num_tensor_parallel_workers
        # batch size is always 1 for prefill and can be different for decode
        self._batch_size = batch_size
        self._prefill_chunk_size = prefill_chunk_size
        self._kv_cache_size = kv_cache_size
        self._max_context_len = max_context_len
        self._is_prefill = is_prefill

        if is_prefill:
            assert batch_size == 1
        else:
            assert prefill_chunk_size == 0

        self._attn_base = 10000 # default from vllm
        self._max_position = 8192 # default from vllm

        self.attn = PagedAttentionWithRoPE(
            self._n_worker_q_heads,
            self._head_dim,
            self._scaling,
            base=self._attn_base,
            max_position=self._max_position,
            rotary_dim=self._head_dim,
            num_kv_heads=self._n_worker_kv_heads,
        ).to(dtype=torch.float16).cuda().eval()
            # .to(dtype=torch.float16).cuda()

        self._blocks_per_sequence = ceil(max_context_len / block_size)
        self._total_num_blocks = max(10000, 1 + batch_size * self._blocks_per_sequence)
        self._k_cache_split_factor = 16 // torch.tensor([], dtype=torch.float16).element_size()

    def _get_input_tensors(self):
        # positions: shape = [num_tokens]
        # query: shape = [num_tokens, num_heads * head_size]
        # key: shape = [num_tokens, num_kv_heads * head_size]
        # value: shape = [num_tokens, num_kv_heads * head_size]
        # key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
        #     block_size, x]
        # value_cache: shape = [num_blocks, num_kv_heads, head_size,
        #     block_size]
        # input_metadata: metadata for paged attention.
        # cache_event: event to wait for the cache operations to finish.
        num_tokens = self._prefill_chunk_size if self._is_prefill else self._batch_size
        num_tokens_padded = _pad_length(num_tokens, 8)

        positions = torch.randint(0, self._max_position, (num_tokens, ), device="cuda")

        query = torch.randn(
            num_tokens,
            self._n_worker_q_heads * self._head_dim,
            dtype=torch.float16,
            device="cuda")
        key = torch.randn(
            num_tokens,
            self._n_worker_kv_heads * self._head_dim,
            dtype=torch.float16,
            device="cuda")
        value = torch.randn(
            num_tokens,
            self._n_worker_kv_heads * self._head_dim,
            dtype=torch.float16,
            device="cuda")
        key_cache = torch.randn(
            (
                self._total_num_blocks,
                self._n_worker_kv_heads,
                self._head_dim // self._k_cache_split_factor,
                self._block_size,
                self._k_cache_split_factor
            ),
            device="cuda",
            dtype=torch.float16
        )
        value_cache = torch.randn(
            (
                self._total_num_blocks,
                self._n_worker_kv_heads,
                self._head_dim,
                self._block_size
            ),
            device="cuda",
            dtype=torch.float16
        )
        block_tables = torch.randint(
            self._total_num_blocks,
            (num_tokens_padded, self._blocks_per_sequence),
            device="cuda",
            dtype=torch.int32
        )
        prefix_plus_current_prompt_tokens_slot_mapping = torch.randint(
            self._total_num_blocks,
            (num_tokens + self._kv_cache_size,),
            device="cuda",
            dtype=torch.int32
        )
        current_tokens_slot_mapping = torch.randint(
            self._total_num_blocks,
            (num_tokens_padded,),
            device="cuda",
            dtype=torch.int32
        )
        context_lens_list = [] if self._is_prefill else [self._kv_cache_size] * self._batch_size
        context_lens_list = _pad_to_alignment(context_lens_list, multiple_of=8)
        context_lens = torch.tensor(
            context_lens_list,
            device="cuda",
            dtype=torch.int32,
        )

        input_metadata = InputMetadata(
            self._prefill_chunk_size,
            self._kv_cache_size,
            self._batch_size,
            context_lens,
            block_tables,
            prefix_plus_current_prompt_tokens_slot_mapping,
            current_tokens_slot_mapping,
        )

        return positions, query, key, value, key_cache, value_cache, input_metadata

    def _run_attention(
        self,
        positions,
        query,
        key,
        value,
        key_cache,
        value_cache,
        input_metadata,):
        # positions: shape = [num_tokens]
        #     query: shape = [num_tokens, num_heads * head_size]
        # key: shape = [num_tokens, num_kv_heads * head_size]
        # value: shape = [num_tokens, num_kv_heads * head_size]
        # key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
        #     block_size, x]
        # value_cache: shape = [num_blocks, num_kv_heads, head_size,
        #     block_size]
        # input_metadata: metadata for paged attention.
        # cache_event: event to wait for the cache operations to finish.

        torch.cuda.synchronize()
        self.attn.forward(
            positions,
            query,
            key,
            value,
            key_cache,
            value_cache,
            input_metadata,
            None,
        )
        torch.cuda.synchronize()

    def profile(self):
        input_tensors = self._get_input_tensors()

        try:
            for _ in range(WARMUP_STEPS):
                self._run_attention(*input_tensors)

            TimerStatsStore.clear_stats()

            for _ in range(ACTIVE_STEPS):
                self._run_attention(*input_tensors)

            return {
                "time_stats": TimerStatsStore.get_stats(),
                "batch_size": self._batch_size,
                "n_embd": self._n_embd,
                "n_q_head": self._n_q_head,
                "n_kv_head": self._n_kv_head,
                "block_size": self._block_size,
                "num_tensor_parallel_workers": self._num_tensor_parallel_workers,
                "prefill_chunk_size": self._prefill_chunk_size,
                "kv_cache_size": self._kv_cache_size,
            }
        except Exception as e:
            print(f"Exception: {e} in exeriment: batch_size: {self._batch_size}, n_embd: {self._n_embd}, "
                  f"n_q_head: {self._n_q_head}, n_kv_head: {self._n_kv_head}, block_size: {self._block_size}, "
                  f"num_tensor_parallel_workers: {self._num_tensor_parallel_workers}, prefill_chunk_size: {self._prefill_chunk_size}, "
                  f"kv_cache_size: {self._kv_cache_size}"
                )
            traceback.print_exc()
            return None
