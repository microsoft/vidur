import binascii
import enum
from itertools import product
from typing import List

from simulator.profiling.attention.attention_input import AttentionInput
from simulator.profiling.collectives.collectives_input import CollectivesInput


class ProfileMethod(enum.Enum):
    CUDA_EVENT = "cuda_event"
    KINETO = "kineto"
    PERF_COUNTER = "perf_counter"
    RECORD_FUNCTION = "record_function"


def get_num_tokens_to_profile(
    max_num_tokens: int,
):
    NUM_TOKENS_SPACE = (
        list([1, 2, 4])
        + list(range(8, 1024, 8))
        + list(range(1024, 2 * 1024 + 1, 16))
        + list(range(2 * 1024, 4 * 1024 + 1, 32))
        + list(range(4 * 1024, 8 * 1024 + 1, 64))
        + list(range(8 * 1024, 16 * 1024 + 1, 128))
        + list(range(16 * 1024, 32 * 1024 + 1, 256))
        + list(range(32 * 1024, 64 * 1024 + 1, 512))
        + list(range(64 * 1024, 128 * 1024 + 1, 1024))
    )
    num_tokens_to_profile = []
    for num_tokens in NUM_TOKENS_SPACE:
        if num_tokens <= max_num_tokens:
            num_tokens_to_profile.append(num_tokens)
        else:
            break
    num_tokens_to_profile.sort(reverse=True)

    return num_tokens_to_profile


def get_attention_batch_sizes_to_profile(min_batch_size: int, max_batch_size: int):
    return list(range(min_batch_size, max_batch_size + 1))


def get_attention_prefill_chunk_sizes_to_profile(max_seq_len: int):
    PREFILL_CHUNK_SIZE_SPACE = (
        [0]
        + list(range(64, 128 + 1, 16))
        + list(range(128, 1024 + 1, 32))
        + list(range(1024, 4 * 1024 + 1, 64))
        + list(range(4 * 1024, 16 * 1024 + 1, 128))
        + list(range(16 * 1024, 64 * 1024 + 1, 256))
    )
    prefill_chunk_sizes_to_profile = []
    for prefill_chunk_size in PREFILL_CHUNK_SIZE_SPACE:
        if prefill_chunk_size <= max_seq_len:
            prefill_chunk_sizes_to_profile.append(prefill_chunk_size)
        else:
            break
    return prefill_chunk_sizes_to_profile


def get_kv_cache_sizes_to_profile(max_seq_len: int):
    KV_CACHE_SIZE_SPACE = (
        list(range(0, 1024 + 1, 32))
        + list(range(1024, 4 * 1024 + 1, 64))
        + list(range(4 * 1024, 64 * 1024 + 1, 256))
        # + list(range(16 * 1024, 64 * 1024 + 1, 512))
    )
    kv_cache_sizes_to_profile = []
    for kv_cache_size in KV_CACHE_SIZE_SPACE:
        if kv_cache_size < max_seq_len:
            kv_cache_sizes_to_profile.append(kv_cache_size)
        else:
            break
    return kv_cache_sizes_to_profile


def get_attention_input_combinations(
    max_seq_len: int,
    min_batch_size: int,
    max_batch_size: int,
    profile_only_prefill: bool,
    profile_only_decode: bool,
):
    prefill_chunk_sizes_to_profile = get_attention_prefill_chunk_sizes_to_profile(
        max_seq_len
    )
    kv_cache_sizes_to_profile = get_kv_cache_sizes_to_profile(max_seq_len)
    batch_sizes_to_profile = get_attention_batch_sizes_to_profile(
        min_batch_size, max_batch_size
    )
    is_prefill_to_profile = [True, False]

    input_combinations = product(
        prefill_chunk_sizes_to_profile,
        kv_cache_sizes_to_profile,
        batch_sizes_to_profile,
        is_prefill_to_profile,
    )

    valid_input_combinations = []

    for input_combination in input_combinations:
        prefill_chunk_size, kv_cache_size, batch_size, is_prefill = input_combination

        if is_prefill and profile_only_decode:
            continue

        if not is_prefill and profile_only_prefill:
            continue

        attention_input = AttentionInput(
            prefill_chunk_size,
            kv_cache_size,
            batch_size,
            is_prefill,
        )

        if attention_input.is_valid(max_seq_len):
            valid_input_combinations.append(attention_input)

    return valid_input_combinations


def get_collectives_sizes_to_profile(max_collective_size: int):
    COLLECTIVE_SIZE_SPACE = (
        list(range(1024, 512 * 1024 + 1, 4 * 1024))
        + list(range(512 * 1024, 8 * 1024 * 1024 + 1, 16 * 1024))
        + list(range(8 * 1024 * 1024, 64 * 1024 * 1024 + 1, 64 * 1024))
        + list(range(64 * 1024 * 1024 + 1, 512 * 1024 * 1024 + 1, 265 * 1024))
    )
    collectives_size_to_profile = []
    for collectives_size in COLLECTIVE_SIZE_SPACE:
        if collectives_size <= max_collective_size:
            collectives_size_to_profile.append(collectives_size)
        else:
            break
    return collectives_size_to_profile


def get_collectives_inputs(
    num_nodes: int,
    num_workers_per_node_combinations: List[int],
    max_collective_size: int,
    collective: str,
    total_gpus_available: int,
):
    num_workers = []

    for num_workers_per_node in num_workers_per_node_combinations:
        for _num_nodes in range(1, num_nodes + 1):
            num_workers.append(num_workers_per_node * _num_nodes)

    num_workers = list(set(num_workers))
    collectives_sizes = get_collectives_sizes_to_profile(max_collective_size)

    collectives_inputs = []

    for num_workers, num_workers_per_node, collective_size in product(
        num_workers, num_workers_per_node_combinations, collectives_sizes
    ):
        collectives_input = CollectivesInput(
            num_workers, num_workers_per_node, collective_size, collective
        )
        if not collectives_input.is_valid(total_gpus_available, num_nodes):
            continue

        collectives_inputs.append(collectives_input)

    return collectives_inputs


def get_cpu_overhead_batch_sizes_to_profile(max_batch_size: int):
    BATCH_SIZE_SPACE = list(range(8, 64 + 1, 8)) + list(range(64, 256 + 1, 16))
    batch_size_to_profile = []
    for batch_size in BATCH_SIZE_SPACE:
        if batch_size <= max_batch_size:
            batch_size_to_profile.append(batch_size)
        else:
            break
    return batch_size_to_profile


def hex_to_binary(hex_identifier):
    return binascii.unhexlify(hex_identifier)
