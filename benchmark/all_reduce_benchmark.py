import datetime
import itertools
import os
import random
import gc
import time
from tqdm import tqdm

import ray
from torch.backends import cudnn
import pandas as pd

from benchmark.all_reduce_wrapper import AllReduceWrapper


NUM_GPUS = 4
OUTPUT_DIR = "all_reduce_benchmarking_output"

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

# [llama 7b, llama 13b, llama 33b, llama 65b, gpt3 175b, code-llama 34b, llama 2 70b, falcon 7b, falcon 40b, falcon 180b]
# lamma 65b, code-llama 34b, laamma 2 70b, and falcon 40b share the same embedding dim
# so we only run one of them
# EMBEDDING_DIMS = [4096, 5120, 6656, 8192, 12288, 4544, 14848]
EMBEDDING_DIMS = [4096, 5120, 6656, 8192]
# EMBEDDING_DIMS = [8192]
NUM_TOKENS = \
    list(range(0, 128, 8)) + \
    list(range(128, 1536, 8)) + \
    list(range(1536, 98 * 1024, 256))
# + \
    # list(range(98 * 1024, 196 * 1024, 512))
# NUM_TOKENS = (
#     list(range(1, 128, 1))
#     + list(range(128, 1536, 4))
#     + list(range(1536, 5 * 1024, 256))
# )
# NUM_TENSOR_PARALLEL_WORKERS = 16
NUM_TENSOR_PARALLEL_WORKERS = [2, 4]
# NUM_TENSOR_PARALLEL_WORKERS = [4]


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

@ray.remote(num_gpus=1)
class ModelRunner:
    def run_all_reduce(
        self, rank, num_workers, comm_id, n_embd, num_tokens
    ):
        # This env var set by Ray causes exceptions with graph building.
        os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
        # Required for properly capturing nccl ops
        os.environ["NCCL_GRAPH_MIXING_SUPPORT"] = "0"

        wrapper = AllReduceWrapper(rank, num_workers, comm_id, n_embd, num_tokens)
        stats = wrapper.profile()
        return stats

    def get_gpu_id(self):
        return ray.get_gpu_ids()[0]


def run_benchmark():
    # create a dir in out dir with human readable timestamp
    output_dir = f"{OUTPUT_DIR}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    runner_pool = []

    used_comm_ids = set()

    ray.init()

    for n_workers in NUM_TENSOR_PARALLEL_WORKERS:
        params = itertools.product(EMBEDDING_DIMS, NUM_TOKENS)

        del runner_pool
        gc.collect()

        time.sleep(2)

        runner_pool = [ModelRunner.remote() for _ in range(n_workers)]
        gpu_ids = safe_ray_get([runner.get_gpu_id.remote() for runner in runner_pool])

        assert sorted(gpu_ids) == list(range(n_workers)), f"gpu_ids: {gpu_ids}"

        for (
            n_embd,
            num_tokens,
        ) in tqdm(list(params)):
            # for each experiment generate a random master port
            while True:
                comm_id = random.randint(65535, 655350000000)
                if comm_id not in used_comm_ids:
                    used_comm_ids.add(comm_id)
                    break

            promises = []
            for rank in range(n_workers):
                promise = runner_pool[rank].run_all_reduce.remote(
                    rank, n_workers, comm_id, n_embd, num_tokens
                )
                promises.append(promise)

            for rank in range(n_workers):
                result = safe_ray_get([promises[rank]])[0]
                if result and rank == 0:
                    all_results.append(result)
                if not result:
                    del runner_pool
                    gc.collect()

                    time.sleep(2)

                    runner_pool = [ModelRunner.remote() for _ in range(n_workers)]
                    gpu_ids = safe_ray_get([runner.get_gpu_id.remote() for runner in runner_pool])

                    assert sorted(gpu_ids) == list(range(n_workers)), f"gpu_ids: {gpu_ids}"

    df = pd.DataFrame(all_results)
    # the time_stats column is a dict, so we need to expand it into columns recursively and add prefix

    df = pd.json_normalize(df["time_stats"]).add_prefix("time_stats.").join(df.drop(columns=["time_stats"]))

    # write results to a csv file
    df.to_csv(f"{output_dir}/results.csv")


if __name__ == "__main__":
    cudnn.benchmark = True
    run_benchmark()
