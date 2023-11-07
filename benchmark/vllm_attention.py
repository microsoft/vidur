"""Multi-head attention."""
from typing import Optional, Tuple

import torch
from torch import nn
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import (
    BlockDiagonalCausalFromBottomRightMask,
)

from vllm import attention_ops
from vllm import cache_ops
from vllm import pos_encoding_ops

from benchmark.cuda_timer import CudaTimer

_SUPPORTED_HEAD_SIZES = [64, 80, 96, 112, 128, 256]
# Should be the same as PARTITION_SIZE in `paged_attention_v2_launcher`.
_PARTITION_SIZE = 512


class InputMetadata:
    def __init__(
        self,
        current_prompt_size: int,
        kv_cache_length: int,
        batch_size: int,
        context_lens: torch.Tensor,
        block_tables : torch.Tensor,
        prefix_plus_current_prompt_tokens_slot_mapping: torch.Tensor,
        current_tokens_slot_mapping: torch.Tensor,
    ) -> None:
        assert batch_size == 1 or current_prompt_size == 0

        self.attn_bias = []
        self.current_prompt_chunk_lens = [current_prompt_size] * batch_size
        self.processed_prompt_lens = [kv_cache_length] * batch_size
        self.block_tables = block_tables
        self.context_lens = context_lens
        self.max_context_len = kv_cache_length
        self.num_current_prompt_tokens = sum(self.current_prompt_chunk_lens)
        self.num_processed_prompt_tokens = sum(self.processed_prompt_lens)
        self.prefix_plus_current_prompt_tokens_slot_mapping = \
            prefix_plus_current_prompt_tokens_slot_mapping
        self.num_valid_tokens = current_tokens_slot_mapping.shape[0]
        self.current_tokens_slot_mapping = current_tokens_slot_mapping
        self.num_generation_tokens = self.context_lens.shape[0]


class PagedAttention(nn.Module):
    # pylint: disable=line-too-long
    """GPT-style multi-head PagedAttention.

    This class takes flattened 1D query, key, and value tensors as input. The
    input 1D tensors can either contain prompt tokens or generation tokens, in
    addition to paddings.

    If the input tensors contain prompt tokens, the layout is as follows:

    |<---------------------- num_valid_tokens ---------------------->|
    |<--------------- num_prompt_tokens -------------->|
    |<--prompt_0-->|<--prompt_1-->|...|<--prompt_N-1-->|<--padding-->|

    Otherwise, the layout is as follows:

    |<------------------ num_valid_tokens ------------------->|
    |<------- num_generation_tokens (M) ------->|
    |<--generation_0-->|...|<--generation_M-1-->|<--padding-->|

    The prompts might have different lengths, while the generation tokens always
    have length 1. The paddings are appended to make the input length a multiple
    of 8, which is desirable for Tensor Cores.

    The class does the following:
    1. Perform multi_query_kv_attention for the prompts. This operation does
        not use the KV cache.
    2. Wait for the cache operations (e.g., swap, copy) to finish. The cache
        operations are issued by the cache engine before executing the forward
        pass of the model, and they are executed asynchronously.
    3. Reshape and store the input key and value tensors in the KV cache.
    4. Perform single_query_cached_kv_attention for the generation tokens.
        This operation reads the previous key and value tensors from the KV
        cache.
    5. Output a flattened 1D tensor.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.head_mapping = torch.repeat_interleave(
            torch.arange(self.num_kv_heads, dtype=torch.int32, device="cuda"),
            self.num_queries_per_kv,
        )

        if self.head_size not in _SUPPORTED_HEAD_SIZES:
            raise ValueError(f"head_size ({self.head_size}) is not supported. "
                             f"Supported head sizes: {_SUPPORTED_HEAD_SIZES}.")

        self._attn_prefill_qkv_reshape_timer = CudaTimer("attn_prefill_qkv_reshape")
        self._attn_prefill_timer = CudaTimer("attn_prefill")
        self._attn_prefill_output_reshape_copy_timer = CudaTimer(
            "attn_prefill_output_reshape_copy")
        self._attn_decode_timer = CudaTimer("attn_decode")
        self._attn_kv_cache_save_timer = CudaTimer(
            "attn_kv_cache_save")
        self._attn_prefill_kv_cache_prep_timer = CudaTimer(
            "attn_prefill_kv_cache_prep")

    def set_attn_bias(
        self,
        input_metadata: InputMetadata,
        dtype: torch.dtype,
    ) -> None:
        del dtype  # Unused.
        if input_metadata.attn_bias:
            # Already set by a previous layer.
            return
        current_prompt_chunk_lens = input_metadata.current_prompt_chunk_lens
        processed_prompt_lens = input_metadata.processed_prompt_lens
        prompt_lens = [
            x + y
            for x, y in zip(current_prompt_chunk_lens, processed_prompt_lens)
        ]
        attn_bias = BlockDiagonalCausalFromBottomRightMask.from_seqlens(
            current_prompt_chunk_lens, prompt_lens)
        input_metadata.attn_bias.append(attn_bias)

    def multi_query_kv_attention(
        self,
        output: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        """Normal attention for the prompt tokens.

        Args:
            output: shape = [num_prompt_tokens, num_heads, head_size]
            query: shape = [num_prompt_tokens, num_heads, head_size]
            key: shape = [num_prompt_tokens, num_kv_heads, head_size]
            value: shape = [num_prompt_tokens, num_kv_heads, head_size]
            input_metadata: metadata for paged attention.
        """
        with self._attn_prefill_qkv_reshape_timer:
            num_prompt_tokens = query.shape[0]
            num_kv_tokens = key.shape[0]
            num_head_groups = self.num_heads // self.num_kv_heads

            query_reshaped = query.reshape([
                1,
                num_prompt_tokens,
                self.num_kv_heads,
                num_head_groups,
                self.head_size,
            ])
            key_reshaped = (key.reshape(
                [1, num_kv_tokens, self.num_kv_heads, 1,
                 self.head_size]).expand([
                     1,
                     num_kv_tokens,
                     self.num_kv_heads,
                     num_head_groups,
                     self.head_size,
                 ]))
            value_reshaped = (value.reshape(
                [1, num_kv_tokens, self.num_kv_heads, 1,
                 self.head_size]).expand([
                     1,
                     num_kv_tokens,
                     self.num_kv_heads,
                     num_head_groups,
                     self.head_size,
                 ]))

        with self._attn_prefill_timer:
            out = xops.memory_efficient_attention_forward(
                query_reshaped,
                key_reshaped,
                value_reshaped,
                attn_bias=input_metadata.attn_bias[0],
                p=0.0,
                scale=self.scale,
            )
        with self._attn_prefill_output_reshape_copy_timer:
            # TODO(woosuk): Unnecessary copy. Optimize.
            output.copy_(
                out.reshape(
                    [num_prompt_tokens, self.num_heads, self.head_size]))
        return output

    def single_query_cached_kv_attention(
        self,
        output: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> None:
        """PagedAttention for the generation tokens.

        Args:
            output: shape = [num_generation_tokens, num_heads, head_size]
            query: shape = [num_generation_tokens, num_heads, head_size]
            key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
                block_size, x]
            value_cache: shape = [num_blocks, num_kv_heads, head_size,
                block_size]
            input_metadata: metadata for paged attention.
        """
        block_size = value_cache.shape[3]
        num_seqs, num_heads, head_size = query.shape
        max_num_partitions = (
            (input_metadata.max_context_len + _PARTITION_SIZE - 1) //
            _PARTITION_SIZE)
        # NOTE(woosuk): We use a simple heuristic to decide whether to use
        # PagedAttention V1 or V2. If the number of partitions is 1, we use
        # V1 to avoid the overhead of reduction. Also, if the number of
        # sequences or heads is large, we use V1 since there is enough work
        # to parallelize.
        # TODO(woosuk): Tune this heuristic.
        use_v1 = max_num_partitions == 1 or num_seqs * num_heads > 512
        if use_v1:
            # Run PagedAttention V1.
            attention_ops.paged_attention_v1(
                output,
                query,
                key_cache,
                value_cache,
                self.head_mapping,
                self.scale,
                input_metadata.block_tables,
                input_metadata.context_lens,
                block_size,
                input_metadata.max_context_len,
                None, # alibi_slopes,
            )
        else:
            # Run PagedAttention V2.
            assert _PARTITION_SIZE % block_size == 0
            tmp_output = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions, head_size),
                dtype=output.dtype,
                device=output.device,
            )
            exp_sums = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions),
                dtype=torch.float32,
                device=output.device,
            )
            max_logits = torch.empty_like(exp_sums)
            attention_ops.paged_attention_v2(
                output,
                exp_sums,
                max_logits,
                tmp_output,
                query,
                key_cache,
                value_cache,
                self.head_mapping,
                self.scale,
                input_metadata.block_tables,
                input_metadata.context_lens,
                block_size,
                input_metadata.max_context_len,
                None, # alibi_slopes,
            )

    def prepare_prefill_kv_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_prompt_tokens = input_metadata.num_current_prompt_tokens
        num_processed_prompt_tokens = input_metadata.num_processed_prompt_tokens

        if num_processed_prompt_tokens > 0:
            # We have already saved the current chunks's KV in the cache.
            # So, we read and return it
            key_current_plus_prev_chunk = torch.empty(
                (num_processed_prompt_tokens + num_prompt_tokens,
                 self.num_kv_heads, self.head_size),
                device=key.device,
                dtype=key.dtype,
            )
            value_current_plus_prev_chunk = torch.empty(
                (num_processed_prompt_tokens + num_prompt_tokens,
                 self.num_kv_heads, self.head_size),
                device=key.device,
                dtype=key.dtype,
            )
            assert (key_cache is not None and value_cache is not None)
            # Wait until the cache op is done.
            if cache_event is not None:
                cache_event.wait()

            cache_ops.gather_cached_kv(
                key_current_plus_prev_chunk,
                value_current_plus_prev_chunk,
                key_cache,
                value_cache,
                input_metadata.prefix_plus_current_prompt_tokens_slot_mapping,
            )
            return key_current_plus_prev_chunk, value_current_plus_prev_chunk
        else:
            return key[:num_prompt_tokens], value[:num_prompt_tokens]

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        """PagedAttention forward pass.

        NOTE: The query, key, and value tensors must be sliced from a qkv
        tensor of shape [num_tokens, 3 * num_heads * head_size].

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
                block_size, x]
            value_cache: shape = [num_blocks, num_kv_heads, head_size,
                block_size]
            input_metadata: metadata for paged attention.
            cache_event: event to wait for the cache operations to finish.

        Returns:
            shape = [num_tokens, num_heads * head_size]
        """

        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        # Pre-allocate the output tensor.
        output = torch.empty_like(query)

        with self._attn_kv_cache_save_timer:
            # Reshape the keys and values and store them in the cache.
            # When key_cache and value_cache are not provided, the new key
            # and value vectors will not be cached.
            num_valid_tokens = input_metadata.num_valid_tokens
            if (num_valid_tokens > 0 and key_cache is not None
                    and value_cache is not None):
                # Wait until the cache op is done.
                if cache_event is not None:
                    cache_event.wait()

                # The stride is 3 because the key and value are sliced from qkv.
                cache_ops.reshape_and_cache(
                    key[:num_valid_tokens],
                    value[:num_valid_tokens],
                    key_cache,
                    value_cache,
                    input_metadata.current_tokens_slot_mapping,
                )

        # Compute the attention op for prompts.
        num_current_prompt_tokens = input_metadata.num_current_prompt_tokens
        if num_current_prompt_tokens > 0:
            # Prompt run.
            with self._attn_prefill_kv_cache_prep_timer:
                final_key, final_value = self.prepare_prefill_kv_cache(
                    key,
                    value,
                    key_cache,
                    value_cache,
                    input_metadata,
                    cache_event,
                )
            self.set_attn_bias(input_metadata, dtype=query.dtype)
            self.multi_query_kv_attention(
                output[:num_current_prompt_tokens],
                query[:num_current_prompt_tokens],
                final_key,
                final_value,
                input_metadata,
            )

        with self._attn_decode_timer:
            if input_metadata.num_generation_tokens > 0:
                # Decoding run.
                assert key_cache is not None and value_cache is not None, (
                    "key_cache and value_cache must be provided when "
                    "generating tokens.")
                # Compute the attention op for generation tokens.
                self.single_query_cached_kv_attention(
                    output[num_current_prompt_tokens:num_valid_tokens],
                    query[num_current_prompt_tokens:num_valid_tokens],
                    key_cache,
                    value_cache,
                    input_metadata,
                )

        # Reshape the output tensor.
        # NOTE(woosuk): The output tensor may include paddings.
        return output.view(-1, self.num_heads * self.head_size)


class PagedAttentionWithRoPE(PagedAttention):
    """PagedAttention with rotary embedding."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        rotary_dim: int,
        max_position: int = 8192,
        base: int = 10000,
        num_kv_heads: Optional[int] = None,
        is_neox_style: bool = True,
    ) -> None:
        super().__init__(num_heads, head_size, scale, num_kv_heads)
        self.is_neox_style = is_neox_style

        # Create the cos and sin cache.
        inv_freq = 1.0 / (base**(torch.arange(
            0, rotary_dim, 2, dtype=torch.float, device="cuda") / rotary_dim))
        t = torch.arange(max_position, dtype=torch.float, device="cuda")
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)

        # FIXME(woosuk): This assumes that we configure the default dtype when
        # initializing the model.
        # TODO(woosuk): Make it more robust.
        torch_dtype = torch.get_default_dtype()
        cache = cache.to(torch_dtype)
        # Embedding size: [max_position, rotary_dim]
        self.register_buffer("cos_sin_cache", cache, persistent=False)

        self._attn_rope_timer = CudaTimer("attn_rope")

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        """PagedAttention forward pass with rotary embedding.

        Args:
            positions: shape = [num_tokens]
                        query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
                block_size, x]
            value_cache: shape = [num_blocks, num_kv_heads, head_size,
                block_size]
            input_metadata: metadata for paged attention.
            cache_event: event to wait for the cache operations to finish.

        Returns:
            shape = [num_tokens, num_heads * head_size]
        """

        # Apply rotary embedding to the query and key before passing them
        # to the attention op.
        with self._attn_rope_timer:
            pos_encoding_ops.rotary_embedding(
                positions,
                query,
                key,
                self.head_size,
                self.cos_sin_cache,
                self.is_neox_style,
            )
        return super().forward(
            query,
            key,
            value,
            key_cache,
            value_cache,
            input_metadata,
            cache_event,
        )

