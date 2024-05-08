import argparse
import datetime
import gc
import os
from itertools import product

import pandas as pd
import ray
from tqdm import tqdm

from simulator.profiling.cpu_overhead.cpu_overhead_benchmark_runner import (
    BenchmarkRunner,
)
from simulator.profiling.utils import (
    get_cpu_overhead_batch_sizes_to_profile,
    hex_to_binary,
)


def parse_args():
    parser = argparse.ArgumentParser(description="CPU Overhead Profiling")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="profiling_outputs",
        help="Output directory for profiling results",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[
            "microsoft/phi-2",
            "internlm/internlm-20b",
            "Qwen/Qwen-72B",
            "meta-llama/Llama-2-7b-hf",
            "codellama/CodeLlama-34b-Instruct-hf",
            "meta-llama/Llama-2-70b-hf",
        ],
        help="Models to profile",
    )
    parser.add_argument(
        "--num_tensor_parallel_workers",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Number of tensor parallel workers to profile",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=128,
        help="Maximum batch size to profile",
    )
    args = parser.parse_args()

    args.output_dir = f"{args.output_dir}/cpu_overhead/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(args.output_dir, exist_ok=True)

    return args


def create_runner(
    model_name: str, batch_size: int, tensor_parallel_degree: int, output_dir: str
) -> BenchmarkRunner:
    placement_group_ids = list(ray.util.placement_group_table().keys())
    for placement_group_id in placement_group_ids:
        ray._private.worker.global_worker.core_worker.remove_placement_group(
            ray.PlacementGroupID(hex_to_binary(placement_group_id))
        )

    num_gpus = 0
    if tensor_parallel_degree == 1:
        num_gpus = 1

    runner_class = (
        ray.remote(num_gpus=num_gpus)(BenchmarkRunner)
        .options(runtime_env={"env_vars": {"KINETO_LOG_LEVEL": "5"}})
        .remote
    )

    return runner_class(model_name, batch_size, tensor_parallel_degree, output_dir)


def main():
    args = parse_args()

    results = []

    batch_sizes_to_profile = get_cpu_overhead_batch_sizes_to_profile(
        args.max_batch_size
    )

    input_combos = product(
        args.models, args.num_tensor_parallel_workers, batch_sizes_to_profile
    )

    pbar = tqdm(total=len(list(input_combos)))

    for model_name, tensor_parallel_degree in product(
        args.models, args.num_tensor_parallel_workers
    ):
        for batch_index, batch_size in enumerate(batch_sizes_to_profile):
            try:
                runner = create_runner(
                    model_name, batch_size, tensor_parallel_degree, args.output_dir
                )
                results.append(ray.get(runner.run.remote()))
                del runner
                # trigger garbage collection
                gc.collect()
            except Exception as e:
                print(
                    f"Failed to run {model_name}_{batch_size}_{tensor_parallel_degree} due to {e}"
                )
                # update progress bar
                pbar.update(len(batch_sizes_to_profile) - batch_index)
                break

            pbar.update(1)

    df = pd.DataFrame(results)
    df.to_csv(f"{args.output_dir}/results.csv")
    df.to_json(f"{args.output_dir}/results.json", orient="records")


if __name__ == "__main__":
    main()
