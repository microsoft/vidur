import torch

from benchmark.cuda_timer import CudaTimer
from benchmark.timer_stats_store import TimerStatsStore


WARMUP_STEPS = 5
GRAPH_STEPS = 10
ACTIVE_STEPS = 10


class GraphAllGather:

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_workers: int,
        dtype: torch.dtype = torch.float16,
        disable_graph: bool = False,
    ) -> None:
        self.vocab_size = vocab_size
        self.disable_graph = disable_graph

        self.buffer = torch.empty(
            size=(context_length, vocab_size // num_workers),
            dtype=dtype,
            device='cuda',
        )
        self.gathered_list = [
            torch.empty_like(self.buffer)
            for _ in range(num_workers)
        ]
        if not self.disable_graph:
            self.graph = self._build_graph()

    def _build_graph(self) -> torch.cuda.CUDAGraph:
        # Warm up.
        torch.distributed.all_gather(self.gathered_list, self.buffer)
        torch.cuda.synchronize()

        # Build graph.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            torch.distributed.all_gather(self.gathered_list, self.buffer)
        torch.cuda.synchronize()
        return graph

    def launch(self) -> torch.Tensor:
        # NOTE: x must be a slice of self.buffer.
        if self.disable_graph:
            torch.distributed.all_reduce(self.buffer)
        else:
            self.graph.replay()


class AllGatherWrapper:
    def __init__(self, rank, num_workers, comm_id, vocab_size, num_tokens):
        self._rank = rank
        self._num_workers = num_workers
        self._vocab_size = vocab_size
        self._num_tokens = num_tokens
        self._comm_id = comm_id

        self._init_communication(comm_id)
        self._graph_all_reduce = GraphAllGather(vocab_size, num_tokens, num_workers, disable_graph=True)

    def _init_communication(self, comm_id):
        print(f"Initializing process group with comm id: {comm_id} for rank: {self._rank} with world size: {self._num_workers}")
        if torch.distributed.is_initialized():
            return

        torch.distributed.init_process_group(
            backend="nccl",
            rank=self._rank,
            world_size=self._num_workers,
            init_method=f"file:///tmp/sing_bm_{comm_id}",
        )

    def _run_all_reduce(self):
        torch.cuda.synchronize()
        with CudaTimer("all_reduce"):
            self._graph_all_reduce.launch()

        torch.cuda.synchronize()

    def profile(self):
        for _ in range(WARMUP_STEPS):
            self._run_all_reduce()

        TimerStatsStore.clear_stats()

        for _ in range(ACTIVE_STEPS):
            self._run_all_reduce()

        return {
            "time_stats": TimerStatsStore.get_stats(),
            "rank": self._rank,
            "num_workers": self._num_workers,
            "vocab_size": self._vocab_size,
            "num_tokens": self._num_tokens,
        }
