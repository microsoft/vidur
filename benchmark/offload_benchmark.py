import datetime
import itertools
import os
import random

import ray
from torch.backends import cudnn
import pandas as pd

from benchmark.offload_wrapper import OffloadWrapper


NUM_GPUS = 8
OUTPUT_DIR = "offload_benchmarking_output"

# NUM_HEADS = [8, 16, 32, 64, 128, 256]
# EMBEDDING_DIMS = [1024, 2048, 4096, 8192, 16384, 32768]
# BATCH_SIZES = [2, 4, 8, 16, 32, 64]
# CONTEXT_LENGTHS = [256, 512, 1024, 2048, 4096, 8192]
# DECODE = [False, True]

# Debug config
# NUM_HEADS = [96]
# EMBEDDING_DIMS = [96 * 128]
# BATCH_SIZES = [2, 4, 8, 16, 32, 64]
# CONTEXT_LENGTHS = [256, 512, 1024, 2048, 4096, 8136]
# DECODE = [True, False]

# GPT-3 config
EMBEDDING_DIMS = [4096, 5120, 6656, 8192, 12288]
BATCH_SIZES = [1, 2, 4]
CONTEXT_LENGTHS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
NUM_TENSOR_PARALLEL_WORKERS = [2, 4, 8]


def safe_ray_get(futures):
    outputs = []
    for future in futures:
        try:
            output = ray.get(future)
            outputs.append(output)
        except Exception as e:
            print(f"Error: {e}")

    return outputs


def run_all_reduce(
    rank, num_workers, master_port, n_embd, batch_size, context_length
):
    wrapper = OffloadWrapper(rank, num_workers, master_port, n_embd, batch_size, context_length)
    stats = wrapper.profile()
    return stats


def run_benchmark():
    run_all_reduce_fn = ray.remote(num_gpus=1)(run_all_reduce)

    # create a dir in out dir with human readable timestamp
    output_dir = f"{OUTPUT_DIR}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    promises = []
    root_rank_promises = []

    params = itertools.product(EMBEDDING_DIMS, BATCH_SIZES, CONTEXT_LENGTHS, NUM_TENSOR_PARALLEL_WORKERS)

    used_ports = set()

    for (
        n_embd,
        batch_size,
        context_length,
        num_workers,
    ) in params:
        if len(promises) + num_workers > NUM_GPUS:
            safe_ray_get(promises)
            all_results.extend(safe_ray_get(root_rank_promises))
            promises = []
            root_rank_promises = []

        # for each experiment generate a random master port
        while True:
            # master_port = random.randint(10000, 65535)
            master_port = random.randint(65535, 655350000000)
            if master_port not in used_ports:
                used_ports.add(master_port)
                break
        # master_port = random.randint(65535, 655350000000)

        for rank in range(num_workers):
            promise = run_all_reduce_fn.remote(
                rank, num_workers, master_port, n_embd, batch_size, context_length
            )
            if rank == 0:
                root_rank_promises.append(promise)
            promises.append(promise)

        # safe_ray_get(promises)
        # all_results.extend(safe_ray_get(root_rank_promises))

    safe_ray_get(promises)
    all_results.extend(safe_ray_get(root_rank_promises))

    df = pd.DataFrame(all_results)
    # the time_stats column is a dict, so we need to expand it into columns recursively and add prefix

    df = pd.json_normalize(df["time_stats"]).add_prefix("time_stats.").join(df.drop(columns=["time_stats"]))

    # write results to a csv file
    df.to_csv(f"{output_dir}/results.csv")


if __name__ == "__main__":
    cudnn.benchmark = True
    run_benchmark()
