import datetime
import itertools
import os
import random
from tqdm import tqdm

import ray
from torch.backends import  cudnn
import pandas as pd

from benchmark.p2p_wrapper import P2PWrapper


NUM_GPUS = 2
OUTPUT_DIR = "p2p_benchmarking_output"

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
# [llama 7b, llama 13b, llama 33b, llama 65b, gpt3 175b, code-llama 34b, llama 2 70b, falcon 7b, falcon 40b, falcon 180b]
# lamma 65b, code-llama 34b, laamma 2 70b, and falcon 40b share the same embedding dim
# so we only run one of them
# EMBEDDING_DIMS = [4096, 5120, 6656, 8192, 12288, 4544, 14848]
EMBEDDING_DIMS = [4096, 5120, 6656, 8192]
NUM_TOKENS = \
    list(range(0, 128, 1)) + \
    list(range(128, 1536, 4)) + \
    list(range(1536, 98 * 1024, 256))
    # + \
    # list(range(98 * 1024, 196 * 1024, 512))
NUM_TENSOR_PARALLEL_WORKERS = [1, 2, 4]


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
    def run_p2p(
        self, rank, comm_id, n_embd, batch_size, context_length,
    ):
        wrapper = P2PWrapper(rank, comm_id, n_embd, batch_size, context_length)
        stats = wrapper.profile()
        return stats


def run_benchmark():
    runner_pool = [ModelRunner.remote() for _ in range(NUM_GPUS)]

    # create a dir in out dir with human readable timestamp
    output_dir = f"{OUTPUT_DIR}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    used_comm_ids = set()

    params = itertools.product(EMBEDDING_DIMS, NUM_TOKENS, NUM_TENSOR_PARALLEL_WORKERS)

    for (
        n_embd,
        num_tokens,
        num_tensor_parallel_workers,
    ) in tqdm(list(params)):
        # for each experiment generate a random master port
        while True:
            # tcp port range is 0-65535
            comm_id = random.randint(10000, 65535)
#            comm_id = random.randint(65535, 655350000000)
            if comm_id not in used_comm_ids:
                used_comm_ids.add(comm_id)
                break

        promises = []
        for rank in range(2):
            # print(f"Running experiment with rank {rank}, comm_id {comm_id}, n_embd {n_embd}, num_tokens {num_tokens}, num_tensor_parallel_workers {num_tensor_parallel_workers}")
            promise = runner_pool[rank].run_p2p.remote(
                rank, comm_id, n_embd, num_tokens, num_tensor_parallel_workers,
            )
            promises.append(promise)

        for rank in range(2):
            result = safe_ray_get([promises[rank]])[0]
            if result and rank == 0:
                all_results.append(result)
            if not result:
                runner_pool[rank] = ModelRunner.remote()

    df = pd.DataFrame(all_results)
    # the time_stats column is a dict, so we need to expand it into columns recursively and add prefix

    df = pd.json_normalize(df["time_stats"]).add_prefix("time_stats.").join(df.drop(columns=["time_stats"]))

    # write results to a csv file
    df.to_csv(f"{output_dir}/results.csv")


if __name__ == "__main__":
    cudnn.benchmark = True
    run_benchmark()
