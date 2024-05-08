import json
import os

import numpy as np
import torch
from sarathi.model_executor.weight_utils import initialize_dummy_weights

from simulator.profiling.mlp.mlp_impl import GPTModel
from simulator.profiling.model_config import ModelConfig
from simulator.profiling.timer_stats_store import TimerStatsStore
from simulator.profiling.utils import ProfileMethod, ProfileOrder

WARMUP_STEPS = 2
ACTIVE_STEPS = 20


class MlpWrapper:
    def __init__(
        self,
        model_config: ModelConfig,
        num_tensor_parallel_workers: int,
        profile_method: ProfileMethod,
        profile_order: ProfileOrder,
        rank: int,
        output_dir: str,
    ):
        super().__init__()

        self.timer_stats_store = TimerStatsStore(profile_method=profile_method)

        self.model_config = model_config
        self.num_tensor_parallel_workers = num_tensor_parallel_workers
        self.profile_method = profile_method
        self.profile_order = profile_order
        self.rank = rank
        self.output_dir = output_dir
        os.makedirs(f"{self.output_dir}/profiler_traces/", exist_ok=True)

        self.model = GPTModel(
            model_config,
            num_tensor_parallel_workers,
            ACTIVE_STEPS if self.profile_method == ProfileMethod.RECORD_FUNCTION else 1,
        )
        initialize_dummy_weights(self.model)
        self.model = self.model.to(dtype=torch.float16).cuda().eval()

    @torch.inference_mode()
    def profile(self, num_tokens: int):
        vocab_range = self.model_config.vocab_size // self.num_tensor_parallel_workers
        input_ids = torch.randint(
            low=0,
            high=vocab_range,
            size=(num_tokens,),
            device="cuda",
            dtype=torch.long,
        )
        positions = torch.arange(num_tokens, device="cuda", dtype=torch.long)
        if self.profile_method == ProfileMethod.RECORD_FUNCTION:
            # Run the model once without capturing the graph.
            # This is to make sure that the captured graph does not include the
            # kernel launches for initial benchmarking (e.g., Triton autotune).
            self.model(
                input_ids,
                positions,
            )
            torch.cuda.synchronize()

            self.timer_stats_store.clear_stats()

            self.start_profiling("cuda")
            self.model(
                input_ids,
                positions,
            )
            self.stop_profiling()
            torch.cuda.synchronize()
            model_hash = str(hash(self.model_config))[:10]
            trace_path = f"{self.output_dir}/profiler_traces/profiler_trace_{model_hash}_num_tokens_{num_tokens}_tp_{self.num_tensor_parallel_workers}.json"
            self.profiler.export_chrome_trace(trace_path)
        else:
            for _ in range(WARMUP_STEPS):
                self.model(
                    input_ids,
                    positions,
                )

            torch.cuda.synchronize()

            self.timer_stats_store.clear_stats()

            for _ in range(ACTIVE_STEPS):
                self.model(
                    input_ids,
                    positions,
                )

            torch.cuda.synchronize()

        stats = {
            "time_stats": (
                self.get_operation_time_stats(trace_path)
                if self.profile_method == ProfileMethod.RECORD_FUNCTION
                else self.timer_stats_store.get_stats()
            ),
            "n_head": self.model_config.num_q_heads,
            "n_kv_head": self.model_config.num_kv_heads,
            "n_embd": self.model_config.embedding_dim,
            "n_expanded_embd": self.model_config.mlp_hidden_dim,
            "vocab_size": self.model_config.vocab_size,
            "use_gated_mlp": self.model_config.use_gated_mlp,
            "num_tokens": num_tokens,
            "num_tensor_parallel_workers": self.num_tensor_parallel_workers,
        }
        self.timer_stats_store.clear_stats()

        return stats

    def start_profiling(self, activity) -> None:
        if activity == "cpu":
            activities = [
                torch.profiler.ProfilerActivity.CPU,
            ]
        elif activity == "cuda":
            activities = [
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        self.profiler = torch.profiler.profile(
            activities=activities,
        )
        self.profiler.__enter__()

    def stop_profiling(self) -> None:
        self.profiler.__exit__(None, None, None)

    def find_children(self, trace, event):
        if not "dur" in event or not "ts" in event:
            return

        children = []
        for e in trace:
            if not "dur" in e or not "ts" in e:
                continue

            # if the ts of the child is completely within the ts of the parent
            if (
                e["ts"] > event["ts"]
                and e["ts"] + e["dur"] < event["ts"] + event["dur"]
            ):
                children.append(e)
        return children

    def find_correlated_event(self, trace, event):
        if not "args" in event or not "correlation" in event["args"]:
            return

        for e in trace:
            if not "args" in e or not "correlation" in e["args"]:
                continue

            if e == event:
                continue

            if e["args"]["correlation"] == event["args"]["correlation"]:
                return e
        return

    def get_operation_time_stats(self, trace_path: str):
        # print("get_operation_time_stats")
        stats = {}

        trace = json.load(open(trace_path, "r"))["traceEvents"]
        for event in trace:
            if not "cat" in event or event["cat"] != "user_annotation":
                continue
            children = self.find_children(trace, event)
            cuda_time = 0
            for child in children:
                if not "cat" in child or child["cat"] != "cuda_runtime":
                    continue
                correlated_event = self.find_correlated_event(trace, child)
                if not correlated_event:
                    continue
                cuda_time += correlated_event["dur"]
            if cuda_time == 0:
                continue

            name = event["name"].replace("vidur_", "")

            if name not in stats:
                stats[name] = []

            stats[name].append(cuda_time * 1e-3)  # to convert to ms

        return {
            operation: {
                "min": np.min(times),
                "max": np.max(times),
                "mean": np.mean(times),
                "median": np.median(times),
                "std": np.std(times),
            }
            for operation, times in stats.items()
        }
