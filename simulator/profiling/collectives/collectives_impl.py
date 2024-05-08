from typing import Callable

import torch

WARMUP_STEPS = 5
GRAPH_STEPS = 3


class GraphedCollective:
    def __init__(
        self,
        num_workers: int,
        size: int,
        collective: str = "all_reduce",
        disable_graph: bool = False,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self._size = size
        self._disable_graph = disable_graph
        self._collective_fn = self._get_collective_fn(collective)

        self._buffer = torch.empty(
            size=(size,),
            dtype=dtype,
            device="cuda",
        )
        self._gather_buffer = None
        if collective == "all_gather":
            self._gather_tensor = torch.empty(
                size=(size * num_workers,),
                dtype=dtype,
                device="cuda",
            )
        elif collective == "reduce_scatter":
            self._reduce_buffer = torch.empty(
                size=(size * num_workers,),
                dtype=dtype,
                device="cuda",
            )

        if not self._disable_graph:
            self._graph = self._build_graph()

    def _run_all_reduce(self):
        torch.distributed.all_reduce(self._buffer)

    def _run_all_gather(self):
        torch.distributed.all_gather_into_tensor(self._gather_tensor, self._buffer)

    def _run_broadcast(self):
        torch.distributed.broadcast(self._buffer, 0)

    def _run_send_recv(self):
        if torch.distributed.get_rank() == 0:
            torch.distributed.send(self._buffer, 1)
        else:
            torch.distributed.recv(self._buffer, 0)

    def _run_reduce_scatter(self):
        torch.distributed.reduce_scatter_tensor(self._buffer, self._reduce_buffer)

    def _get_collective_fn(self, collective: str) -> Callable:
        if collective == "all_reduce":
            return self._run_all_reduce
        elif collective == "all_gather":
            return self._run_all_gather
        elif collective == "broadcast":
            return self._run_broadcast
        elif collective == "send_recv":
            return self._run_send_recv
        elif collective == "reduce_scatter":
            return self._run_reduce_scatter
        else:
            raise ValueError(f"Unknown collective: {collective}")

    def _build_graph(self) -> torch.cuda.CUDAGraph:
        # Warm up.
        for _ in range(WARMUP_STEPS):
            self._collective_fn()

        torch.cuda.synchronize()

        # Build graph.
        graph = torch.cuda.CUDAGraph()

        mempool = torch.cuda.graph_pool_handle()

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
        ):
            with torch.cuda.graph(graph, mempool):
                for _ in range(GRAPH_STEPS):
                    self._collective_fn()

        torch.cuda.synchronize()
        return graph

    def launch(self) -> torch.Tensor:
        # NOTE: x must be a slice of self._buffer.
        if self._disable_graph:
            self._collective_fn()
        else:
            self._graph.replay()
