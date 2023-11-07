import datetime
import itertools
import os

import ray
from torch.backends import cudnn
import pandas as pd
from tqdm import tqdm

from benchmark.mixed_attention_wrapper import MixedAttentionWrapper

NUM_GPUS = 4
OUTPUT_DIR = "mixed_attention_benchmarking_output"

# Debug config
# NUM_HEADS = [96]
# EMBEDDING_DIMS = [96 * 128]
# BATCH_SIZES = [2, 4, 8, 16, 32, 64]
# CONTEXT_LENGTHS = [256, 512, 1024, 2048, 4096, 8136]
# DECODE = [True, False]

# [llama 7b, llama 13b, llama 33b, llama 65b, gpt3 175b, code-llama 34b, llama 2 70b, falcon 7b, falcon 40b, falcon 180b]
# NUM_HEADS = [32, 40, 52, 64, 96, 64, 64, 71, 128, 232]
# NUM_KV_HEAD = [32, 40, 52, 64, 96, 8, 8, 1, 8, 8]
# EMBEDDING_DIMS = [4096, 5120, 6656, 8192, 12288, 8192, 8192, 4544, 8192, 14848]
# NUM_HEADS = [32]
# NUM_KV_HEAD = [32]
# EMBEDDING_DIMS = [4096]
NUM_HEADS = [32, 40, 52, 64, 64]
NUM_KV_HEAD = [32, 40, 52, 64, 8]
EMBEDDING_DIMS = [4096, 5120, 6656, 8192, 8192]
# NUM_HEADS_EMBEDDING_DIMS = list(zip(NUM_HEADS, NUM_KV_HEAD, EMBEDDING_DIMS))
# # BATCH_SIZES = range(1, 65)
# # BATCH_SIZES = [1]
# # PREFILL_CHUNK_SIZES = [1, 128] + list(range(256, 1024 * 4 + 1, 256)) + list(range(1024 * 4 + 512, 1024 * 8 + 1, 512))
# # PREFILL_CHUNK_SIZES = [0] + list(range(1024 * 16, 1024 * 32 + 1, 512))
# # NUM_TENSOR_PARALLEL_WORKERS = [1, 2, 4, 8]
# PREFILL_CHUNK_SIZES = [0]
# NUM_TENSOR_PARALLEL_WORKERS = [1, 2, 4, 8]
# # KV_CACHE_SIZES = list(range(0, 1024 + 1, 64)) + list(range(1024 + 128, 1024 * 4 + 1, 128)) + list(range(1024 * 4 + 512, 1024 * 8 + 1, 512))
# # KV_CACHE_SIZES = list(range(0, 1024 + 1, 64)) + list(range(1024 + 128, 1024 * 4 + 1, 128)) + list(range(1024 * 4 + 512, 1024 * 32 + 1, 512))
# # KV_CACHE_SIZES = [1024]
# KV_CACHE_SIZES = [1024 * 4, 1024 * 2, 1024, 512, 256]
# BATCH_SIZES = [1, 2, 4, 8, 16]
# # IS_PREFILL = [True, False]
# IS_PREFILL = [False]
# # MAX_SEQ_LEN = 1024 * 8
# MAX_SEQ_LEN = 1024 * 8
# BLOCK_SIZE = 16


# NUM_HEADS = [32]
# NUM_KV_HEAD = [32]
# EMBEDDING_DIMS = [4096]
# NUM_HEADS = [64]
# NUM_KV_HEAD = [8]
# EMBEDDING_DIMS = [8192]
NUM_HEADS_EMBEDDING_DIMS = list(zip(NUM_HEADS, NUM_KV_HEAD, EMBEDDING_DIMS))
BATCH_SIZES = range(1, 65)
# BATCH_SIZES = [8]
PREFILL_CHUNK_SIZES = [0] + [128] + list(range(256, 1024 * 4 + 1, 256))
# PREFILL_CHUNK_SIZES = [0] + list(range(1024 * 4 + 512, 1024 * 8 + 1, 512)) + list(range(1024 * 16, 1024 * 32 + 1, 512))
NUM_TENSOR_PARALLEL_WORKERS = [1, 2, 4]
# NUM_TENSOR_PARALLEL_WORKERS = [8]
# PREFILL_CHUNK_SIZES = [0]
# NUM_TENSOR_PARALLEL_WORKERS = [1]
KV_CACHE_SIZES = list(range(0, 1024 + 1, 64)) + list(range(1024 + 128, 1024 * 4 + 1, 128))
# KV_CACHE_SIZES = [2048]
# KV_CACHE_SIZES = list(range(0, 1024 + 1, 64)) + list(range(1024 + 128, 1024 * 4 + 1, 128)) + list(range(1024 * 4 + 512, 1024 * 32 + 1, 512))
# KV_CACHE_SIZES = [1024]
IS_PREFILL = [True, False]
# IS_PREFILL = [False]
MAX_SEQ_LEN = 1024 * 4
BLOCK_SIZE = 16



def safe_ray_get(futures):
    outputs = []
    for future in futures:
        try:
            output = ray.get(future)
            outputs.append(output)
        except Exception as e:
            print(f"Error: {e}")
            outputs.append(None)

    return outputs


# define a ray actor which has run_model function
@ray.remote(num_gpus=1)
class ModelRunner:
    def run_model(
        self,
        batch_size,
        n_head,
        n_kv_head,
        n_embd,
        num_tensor_parallel_workers,
        prefill_chunk_size,
        kv_cache_size,
        is_prefill
    ):
        model = MixedAttentionWrapper(
            batch_size,
            n_head,
            n_kv_head,
            n_embd,
            BLOCK_SIZE,
            num_tensor_parallel_workers,
            prefill_chunk_size,
            kv_cache_size,
            MAX_SEQ_LEN,
            is_prefill,
        )
        stats = model.profile()
        del model
        return stats


def run_benchmark():
    promises = []

    # create a dir in out dir with human readable timestamp
    output_dir = f"{OUTPUT_DIR}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(output_dir,
    exist_ok=True)

    all_results = []

    # create actor pool with NUM_GPUS actors
    model_runners = [ModelRunner.remote() for _ in range(NUM_GPUS)]

    params = list(itertools.product(
        NUM_HEADS_EMBEDDING_DIMS,
        BATCH_SIZES,
        PREFILL_CHUNK_SIZES,
        KV_CACHE_SIZES,
        NUM_TENSOR_PARALLEL_WORKERS,
        IS_PREFILL
    ))

    for (
        n_head_n_embd,
        batch_size,
        prefill_chunk_size,
        kv_cache_size,
        num_tensor_parallel_workers,
        is_prefill,
    ) in tqdm(params):
        if is_prefill and batch_size > 1:
            continue

        if not is_prefill and prefill_chunk_size > 0:
            continue

        if is_prefill and prefill_chunk_size == 0:
            continue

        if prefill_chunk_size + kv_cache_size > MAX_SEQ_LEN:
            continue

        if not is_prefill and kv_cache_size == 0:
            continue

        # if prefill and kv_cache_size % prefill_chunk_size != 0:
        #     continue

        # for falcon-7b
        if n_head_n_embd[0] % num_tensor_parallel_workers != 0:
            continue

        worker_id = len(promises)
        promise = model_runners[worker_id].run_model.remote(
            batch_size,
            n_head_n_embd[0],
            n_head_n_embd[1],
            n_head_n_embd[2],
            num_tensor_parallel_workers,
            prefill_chunk_size,
            kv_cache_size,
            is_prefill
        )
        promises.append((worker_id, promise))

        if len(promises) >= NUM_GPUS:
            for worker_id, promise in promises:
                result = safe_ray_get([promise])[0]
                if result:
                    all_results.append(result)
                else:
                    # replace the worker that failed
                    model_runners[worker_id] = ModelRunner.remote()
            promises = []

    results = safe_ray_get([p for _, p in promises])
    stats = [result for result in results if result]
    all_results.extend(stats)

    df = pd.DataFrame(all_results)
    # the time_stats column is a dict, so we need to expand it into columns recursively and add prefix
    df = pd.json_normalize(df["time_stats"]).add_prefix("time_stats.").join(df.drop(columns=["time_stats"]))

    # write results to a csv file
    df.to_csv(f"{output_dir}/results.csv")


if __name__ == "__main__":
    cudnn.benchmark = True
    run_benchmark()
