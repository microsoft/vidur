import torch

from benchmark.cuda_timer import CudaTimer
from benchmark.timer_stats_store import TimerStatsStore


WARMUP_STEPS = 5
GRAPH_STEPS = 10
ACTIVE_STEPS = 5


class GraphSendRecv:

    def __init__(
        self,
        rank: int,
        n_embd: int,
        context_length: int,
        dtype: torch.dtype = torch.float16,
        disable_graph: bool = False,
    ) -> None:
        self.rank = rank
        self.n_embd = n_embd
        self.disable_graph = disable_graph

        self.buffer = torch.empty(
            size=(context_length, n_embd),
            dtype=dtype,
            device='cuda',
        )
        self.graph = self._build_graph()

    def _run_send_recv(self):
        if self.rank == 0:
            torch.distributed.send(self.buffer, dst=1)
        else:
            torch.distributed.recv(self.buffer, src=0)

    def _build_graph(self) -> torch.cuda.CUDAGraph:
        # Warm up.
        for _ in range(WARMUP_STEPS):
            self._run_send_recv()

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
                    self._run_send_recv()

        torch.cuda.synchronize()
        return graph

    def launch(self) -> torch.Tensor:
        # NOTE: x must be a slice of self.buffer.
        if self.disable_graph:
            self._run_send_recv()
        else:
            self.graph.replay()


class P2PWrapper:
    def __init__(self, rank, comm_id, n_embd, num_tokens, num_tensor_parallel_workers):
        self._rank = rank
        self._n_embd = n_embd
        self._num_tokens = num_tokens
        self._num_tensor_parallel_workers = num_tensor_parallel_workers

        self._init_communication(rank, comm_id)
        self._graph_send_recv = GraphSendRecv(self._rank, n_embd // num_tensor_parallel_workers, num_tokens)

        self._cuda_timer = CudaTimer("send_recv", aggregation_fn=np.median, filter_str="ncclKernel")


    def _init_communication(self, rank, comm_id):
        # skip if already initialized
        if torch.distributed.is_initialized():
            return

        print(f"Initializing process group with comm_id: {comm_id} for rank: {self._rank} at file:///tmp/sing_comm_{comm_id}")
        torch.distributed.init_process_group(
            backend="nccl",
            rank=self._rank,
            world_size=2,
            # init_method=f"tcp://node-0:{comm_id}",
            init_method=f"file:///tmp/sing_comm_{comm_id}",
        )
        print(f"Initialized process group.")

    def _run_send_recv(self):
        torch.cuda.synchronize()
        torch.distributed.barrier()

        with self._cuda_timer:
            self._graph_send_recv.launch()

        torch.cuda.synchronize()

    def profile(self):
        TimerStatsStore.clear_stats()

        for _ in range(ACTIVE_STEPS):
            self._run_send_recv()

        return {
            "time_stats": TimerStatsStore.get_stats(),
            "n_embd": self._n_embd,
            "num_tokens": self._num_tokens,
            "rank": self._rank,
            "num_tensor_parallel_workers": self._num_tensor_parallel_workers,
        }
