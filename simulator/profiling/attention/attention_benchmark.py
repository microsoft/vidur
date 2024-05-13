import argparse
import datetime
import itertools
import os
from typing import Any, List

import pandas as pd
import ray
from sarathi.model_executor.attention import AttentionBackend
from tqdm import tqdm

from simulator.profiling.attention.attention_input import AttentionInput
from simulator.profiling.attention.attention_wrapper import AttentionWrapper
from simulator.profiling.common.model_config import ModelConfig
from simulator.profiling.utils import ProfileMethod, get_attention_input_combinations


def parse_args():
    parser = argparse.ArgumentParser(description="Attention Profiling")
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
        "--max_model_len",
        type=int,
        default=4096,
        help="Maximum context length model can server",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=4096,
        help="Maximum context length of input",
    )
    parser.add_argument(
        "--min_batch_size",
        type=int,
        default=1,
        help="Maximum decode batch size",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=128,
        help="Maximum decode batch size",
    )
    parser.add_argument(
        "--profile_only_decode",
        action="store_true",
        help="Only profile the decode",
    )
    parser.add_argument(
        "--profile_only_prefill",
        action="store_true",
        help="Only profile the prefill",
    )
    parser.add_argument(
        "--profile_method",
        default=ProfileMethod.KINETO,
        const=ProfileMethod.KINETO,
        nargs="?",
        choices=[e.value for e in ProfileMethod],
        help="Method to use for measuring time taken by operations (default: %(default)s)",
    )
    parser.add_argument(
        "--attention_backend",
        default=AttentionBackend.FLASH_ATTENTION,
        const=AttentionBackend.FLASH_ATTENTION,
        nargs="?",
        choices=[e.value for e in AttentionBackend],
        help="The attention backend to profile (default: %(default)s)",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=16,
        help="Block size for paged attention",
    )
    args = parser.parse_args()

    args.output_dir = f"{args.output_dir}/attention/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(args.output_dir, exist_ok=True)

    return args


def profile_model(
    args: argparse.Namespace,
    model: str,
    input_combinations: List[AttentionInput],
    pbar: Any,
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
        AttentionWrapper,
    ).options(runtime_env={"env_vars": {"KINETO_LOG_LEVEL": "5"}})

    for num_tensor_parallel_workers in args.num_tensor_parallel_workers:
        model_wrappers = [
            model_wrapper_actor.remote(
                model_config,
                num_tensor_parallel_workers,
                args.max_model_len,
                args.block_size,
                args.attention_backend,
                args.profile_method,
            )
            for _ in range(args.num_gpus)
        ]

        for attention_input in input_combinations:
            worker_id = len(promises)
            promise = model_wrappers[worker_id].profile.remote(attention_input)
            promises.append(promise)

            if len(promises) >= args.num_gpus:
                results = ray.get(promises)
                all_results.extend(results)
                promises = []

            pbar.update(1)

    results = ray.get(promises)
    all_results.extend(results)

    # filter all none results
    all_results = list(filter(None, all_results))

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

    input_combinations = get_attention_input_combinations(
        args.max_seq_len,
        args.min_batch_size,
        args.max_batch_size,
        args.profile_only_prefill,
        args.profile_only_decode,
    )

    total_combos = itertools.product(
        args.models,
        args.num_tensor_parallel_workers,
        input_combinations,
    )

    pbar = tqdm(total=len(list(total_combos)))

    result_dfs = []

    for model in args.models:
        result_df = profile_model(
            args,
            model,
            input_combinations,
            pbar,
        )
        result_dfs.append(result_df)

    # combine all results into a single dataframe
    combined_df = pd.concat(result_dfs, ignore_index=True)
    # write combined results to a csv file
    combined_df.to_csv(f"{args.output_dir}/combined.csv", index=False)


if __name__ == "__main__":
    main()
