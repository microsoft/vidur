import datetime
import itertools
import os

import ray
import torch
import torch.backends.cudnn as cudnn
import pandas as pd
from tqdm import tqdm

from benchmark.mlp_wrapper import ModelWrapper

USE_RAY = True
NUM_GPUS = 4
OUTPUT_DIR = "benchmarking_outputs"

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
# EXPANDED_EMBEDDING_DIMS = [11008, 13824, 17920, 22016, 49152, 22016, 28672, 18176, 32768, 59392]
# VOCAB_SIZE = [32768, 32768, 32768, 32768, 50257, 32768, 32768, 65024, 65024, 65024]
# USE_GATED_MLP = [True, True, True, True, False, True, True, False, False, False]
NUM_HEADS = [32, 40, 52, 64, 64, 64]
NUM_KV_HEAD = [32, 40, 52, 64, 8, 8]
EMBEDDING_DIMS = [4096, 5120, 6656, 8192, 8192, 8192]
EXPANDED_EMBEDDING_DIMS = [11008, 13824, 17920, 22016, 22016, 28672]
VOCAB_SIZE = [32768, 32768, 32768, 32768, 32768, 32768]
USE_GATED_MLP = [True, True, True, True, True, True]

# NUM_HEADS = [32]
# NUM_KV_HEAD = [32]
# EMBEDDING_DIMS = [4096]
# EXPANDED_EMBEDDING_DIMS = [11008]
# VOCAB_SIZE = [32768]
# USE_GATED_MLP = [True]

# NUM_HEADS = [64]
# NUM_KV_HEAD = [8]
# EMBEDDING_DIMS = [8192]
# EXPANDED_EMBEDDING_DIMS = [28672]
# VOCAB_SIZE = [32768]
# USE_GATED_MLP = [True]


# NUM_HEADS = [64, 64, 71, 128, 232]
# NUM_KV_HEAD = [8, 8, 1, 8, 8]
# EMBEDDING_DIMS = [8192, 8192, 4544, 8192, 14848]
# EXPANDED_EMBEDDING_DIMS = [22016, 28672, 18176, 32768, 59392]
# VOCAB_SIZE = [32768, 32768, 65024, 65024, 65024]
# USE_GATED_MLP = [True, True, False, False, False]

MODEL_CONFIGS = list(zip(NUM_HEADS, NUM_KV_HEAD, EMBEDDING_DIMS, EXPANDED_EMBEDDING_DIMS, VOCAB_SIZE, USE_GATED_MLP))
NUM_TOKENS = (
    list(range(1, 1536, 8))
    + list(range(1536, 98 * 1024, 256))
    # + list(range(98 * 1024, 196 * 1024, 512))
)
# NUM_TOKENS = (
#     list(range(1, 128, 1))
#     + list(range(128, 1536, 4))
#     + list(range(1536, 5 * 1024, 256))
# )
# NUM_TOKENS = [8]
# NUM_TENSOR_PARALLEL_WORKERS = [1, 2, 4, 8]
NUM_TENSOR_PARALLEL_WORKERS = [1, 2, 4]
# NUM_TENSOR_PARALLEL_WORKERS = [8]


def safe_ray_get(futures):
    outputs = []
    for future in futures:
        try:
            output = ray.get(future)
            outputs.append(output)
        except Exception as e:
            print(f"Error: {e}")
            outputs.append((False, None))

    return outputs


class ModelRunner:
    def run_model(
        self,
        model_config,
        num_tokens,
        num_tensor_parallel_workers,
    ):
        model = ModelWrapper(
            model_config,
            num_tokens,
            num_tensor_parallel_workers,
        )
        fits_in_memory = model.profile()
        del model
        return fits_in_memory

    def start_profiling(self) -> None:
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        )
        self.profiler.__enter__()

    def stop_profiling(self, output_dir: str) -> None:
        self.profiler.__exit__(None, None, None)
        print(f"Profiler output dir: {output_dir}")
        print(output_dir)
        self.profiler.export_chrome_trace(f"{output_dir}/profiler_trace.json")


def run_benchmark():
    if USE_RAY:
        # create actor pool with NUM_GPUS actors
        model_runner_actor = ray.remote(num_cpus=1, num_gpus=1)(ModelRunner,)
        model_runners = [model_runner_actor.remote() for _ in range(NUM_GPUS)]
    else:
        model_runner = ModelRunner()

    output_dir = f"{OUTPUT_DIR}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(output_dir, exist_ok=True)

    total_combos = itertools.product(
        MODEL_CONFIGS,
        NUM_TOKENS,
        NUM_TENSOR_PARALLEL_WORKERS,
    )

    pbar = tqdm(total=len(list(total_combos)))

    # model_runners[0].start_profiling.remote()

    for model_config in MODEL_CONFIGS:
        model_name = f"n_head_{model_config[0]}_n_kv_head_{model_config[1]}_n_embd_{model_config[2]}_" \
                     f"n_expanded_embd_{model_config[3]}_vocab_size_{model_config[4]}_use_gated_mlp_{model_config[5]}"
        file_path = f"{output_dir}/results_{model_name}.csv"
        promises = []
        all_results = []
        params = itertools.product(
            NUM_TOKENS,
            NUM_TENSOR_PARALLEL_WORKERS,
        )
    
        for (
            num_tokens,
            num_tensor_parallel_workers,
        ) in list(params):
            if USE_RAY:
                worker_id = len(promises)
                promise = model_runners[worker_id].run_model.remote(
                    model_config,
                    num_tokens,
                    num_tensor_parallel_workers,
                )
                promises.append((worker_id, promise))

                if len(promises) >= NUM_GPUS:
                    for worker_id, promise in promises:
                        status, stat = safe_ray_get([promise])[0]
                        if status:
                            all_results.append(stat)
                        else:
                            # replace the worker that failed
                            model_runners[worker_id] = model_runner_actor.remote()
                    promises = []
            else:
                status, stat = model_runner.run_model(
                    model_config,
                    num_tokens,
                    num_tensor_parallel_workers,
                )

                if status:
                    all_results.append(stat)

            pbar.update(1)

        # model_runners[0].stop_profiling.remote(output_dir)

        if USE_RAY:
            results = safe_ray_get([p for _, p in promises])
            stats = [stats for status, stats in results if status]
            all_results.extend(stats)

        df = pd.DataFrame(all_results)
        # the time_stats column is a dict, so we need to expand it into columns recursively and add prefix
        df = (
            pd.json_normalize(df["time_stats"])
            .add_prefix("time_stats.")
            .join(df.drop(columns=["time_stats"]))
        )
        # write results to a csv file
        df.to_csv(file_path)


if __name__ == "__main__":
    cudnn.benchmark = True
    run_benchmark()
