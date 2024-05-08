import argparse
import datetime
import itertools
import os
from typing import Any, List

import pandas as pd
import ray
import yaml
from tqdm import tqdm

from simulator.profiling.mlp.mlp_wrapper import MlpWrapper
from simulator.profiling.model_config import ModelConfig
from simulator.profiling.utils import (
    ProfileMethod,
    ProfileOrder,
    get_num_tokens_to_profile,
)


def parse_args():
    parser = argparse.ArgumentParser(description="MLP Profiling")
    parser.add_argument(
        "--disable_ray",
        action="store_true",
        help="Disable Ray",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=8,
        help="Number of GPUs to use for profiling",
    )
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
        "--max_tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens to profile",
    )
    parser.add_argument(
        "--profile_method",
        default=ProfileMethod.RECORD_FUNCTION,
        const=ProfileMethod.RECORD_FUNCTION,
        nargs="?",
        choices=[e.value for e in ProfileMethod],
        help="Method to use for measuring time taken by operations (default: %(default)s)",
    )
    parser.add_argument(
        "--profile_order",
        default=ProfileOrder.DECREASING,
        const=ProfileOrder.DECREASING,
        nargs="?",
        choices=[e.value for e in ProfileOrder],
        help="In what order the matmul sizes are profiled (default: %(default)s)",
    )
    args = parser.parse_args()

    args.output_dir = (
        f"{args.output_dir}/mlp/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    return args


def profile_model(
    args: argparse.Namespace, model: str, num_tokens_to_profile: List[int], pbar: Any
):
    model_config = ModelConfig.from_model_name(model)
    output_file_path = f"{args.output_dir}/{model}.csv"
    # create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    promises = []
    all_results = []

    model_wrapper_actor = ray.remote(
        num_cpus=1,
        num_gpus=1,
    )(
        MlpWrapper,
    ).options(runtime_env={"env_vars": {"KINETO_LOG_LEVEL": "5"}})

    for num_tensor_parallel_workers in args.num_tensor_parallel_workers:
        if model_config.no_tensor_parallel and num_tensor_parallel_workers > 1:
            pbar.update(len(num_tokens_to_profile))
            continue

        model_wrappers = [
            model_wrapper_actor.remote(
                model_config,
                num_tensor_parallel_workers,
                args.profile_method,
                args.profile_order,
                rank,
                args.output_dir,
            )
            for rank in range(args.num_gpus)
        ]
        for num_tokens in num_tokens_to_profile:
            worker_id = len(promises)
            promise = model_wrappers[worker_id].profile.remote(
                num_tokens,
            )
            promises.append(promise)

            if len(promises) >= args.num_gpus:
                results = ray.get(promises)
                all_results.extend(results)
                promises = []

            pbar.update(1)

    results = ray.get(promises)
    all_results.extend(results)

    df = pd.DataFrame(all_results)
    # the time_stats column is a dict, so we need to expand it into columns recursively and add prefix
    df = (
        pd.json_normalize(df["time_stats"])
        .add_prefix("time_stats.")
        .join(df.drop(columns=["time_stats"]))
    )
    # write results to a csv file
    df.to_csv(output_file_path, index=False)
    df.to_json(f"{args.output_dir}/{model}.json", orient="records")

    return df


def main():
    args = parse_args()
    yaml.dump(vars(args), open(f"{args.output_dir}/config.yaml", "w"))

    num_tokens_to_profile = get_num_tokens_to_profile(
        args.max_tokens, args.profile_order
    )

    total_combos = itertools.product(
        args.models,
        num_tokens_to_profile,
        args.num_tensor_parallel_workers,
    )

    pbar = tqdm(total=len(list(total_combos)))

    result_dfs = []

    for model in args.models:
        result_df = profile_model(
            args,
            model,
            num_tokens_to_profile,
            pbar,
        )
        result_dfs.append(result_df)

    # combine all results into a single dataframe
    combined_df = pd.concat(result_dfs, ignore_index=True)
    # write combined results to a csv file
    combined_df.to_csv(f"{args.output_dir}/combined.csv", index=False)


if __name__ == "__main__":
    main()
