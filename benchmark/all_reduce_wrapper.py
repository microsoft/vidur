import torch

import numpy as np

from benchmark.cuda_timer import CudaTimer
from benchmark.timer_stats_store import TimerStatsStore

from vllm.all_reduce_ops import init_nccl, all_reduce


WARMUP_STEPS = 5
GRAPH_STEPS = 10
ACTIVE_STEPS = 5
OUTPUT_DIR = "all_reduce_benchmarking_output"
PROFILE = False


class GraphAllReduce:

    def __init__(
        self,
        n_embd: int,
        context_length: int,
        dtype: torch.dtype = torch.float16,
        disable_graph: bool = False,
    ) -> None:
        self.n_embd = n_embd
        self.disable_graph = disable_graph

        self.buffer = torch.empty(
            size=(context_length, n_embd),
            dtype=dtype,
            device='cuda',
        )
        if not self.disable_graph:
            self.graph = self._build_graph()

    def _build_graph(self) -> torch.cuda.CUDAGraph:
        # Warm up.
        for _ in range(WARMUP_STEPS):
            torch.distributed.all_reduce(self.buffer)

        torch.cuda.synchronize()

        # Build graph.
        graph = torch.cuda.CUDAGraph()

        mempool = torch.cuda.graph_pool_handle()

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU],
        ):
            with torch.cuda.graph(graph, mempool):
                for _ in range(GRAPH_STEPS):
                    torch.distributed.all_reduce(self.buffer)
                # all_reduce(self.buffer)

        torch.cuda.synchronize()
        return graph

    def launch(self) -> torch.Tensor:
        # NOTE: x must be a slice of self.buffer.
        if self.disable_graph:
            torch.distributed.all_reduce(self.buffer)
            # all_reduce(self.buffer)
        else:
            self.graph.replay()


class AllReduceWrapper:
    def __init__(self, rank, num_workers, comm_id, n_embd, num_tokens):
        self._rank = rank
        self._num_workers = num_workers
        self._n_embd = n_embd
        self._num_tokens = num_tokens
        self._comm_id = comm_id

        self._init_communication(comm_id)
        self._graph_all_reduce = GraphAllReduce(n_embd, num_tokens, disable_graph=False)

        self._cuda_timer = CudaTimer("all_reduce", aggregation_fn=np.median, filter_str="ncclKernel")

    def _init_communication(self, comm_id):
        if torch.distributed.is_initialized():
            return

        import os
        print(f"Rank: {self._rank}, num_workers: {self._num_workers}, comm_id: {comm_id}")
        print("CUDA_VISIBLE_DEVICES: ", os.environ["CUDA_VISIBLE_DEVICES"])
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        print(f"Initializing process group with comm id: {comm_id} for rank: {self._rank} with world size: {self._num_workers}")
        torch.cuda.set_device(self._rank)

        torch.distributed.init_process_group(
            backend="nccl",
            rank=self._rank,
            world_size=self._num_workers,
            # init_method=f"file:///serenity/scratch/tmp/sing_bm_{comm_id}",
            init_method=f"file:///tmp/sing_bm_{comm_id}",
        )
        # init_nccl(self._num_workers, self._rank, f"/tmp/comm_{comm_id}")

    def _run_all_reduce(self):
        torch.cuda.synchronize()
        torch.distributed.barrier()

        with self._cuda_timer:
            self._graph_all_reduce.launch()

        torch.cuda.synchronize()

    def profile(self):
        TimerStatsStore.clear_stats()
        if PROFILE:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(ACTIVE_STEPS):
                    self._run_all_reduce()
            prof.export_chrome_trace(f"{OUTPUT_DIR}/jid_{self._comm_id}_rank_{self._rank}.json")
        else:
            for _ in range(ACTIVE_STEPS):
                self._run_all_reduce()

        return {
            "time_stats": TimerStatsStore.get_stats(),
            "rank": self._rank,
            "num_workers": self._num_workers,
            "n_embd": self._n_embd,
            "num_tokens": self._num_tokens,
        }
