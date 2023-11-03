import time

import torch

from benchmark.cuda_timer import CudaTimer
from benchmark.timer_stats_store import TimerStatsStore


WARMUP_STEPS = 10
ACTIVE_STEPS = 100
PROFILE = False


class OffloadWrapper:
    def __init__(self, rank, num_workers, master_port, n_embd, batch_size, context_length):
        self._rank = rank
        self._num_workers = num_workers
        self._n_embd = n_embd
        self._batch_size = batch_size
        self._context_length = context_length
        self._master_port = master_port

        self._cpu_buffer = torch.empty(
            size=(2 * batch_size * context_length, n_embd // num_workers),
            dtype=torch.float16,
            device='cpu',
        ).pin_memory()
        self._gpu_buffer = torch.randn(
            size=(2 * batch_size * context_length, n_embd // num_workers),
            dtype=torch.float16,
            device='cuda',
        )

    def _run_all_offload(self):
        torch.cuda.synchronize()
        with CudaTimer("offload"):
            self._cpu_buffer.copy_(self._gpu_buffer, non_blocking=True)
        torch.cuda.synchronize()

    def profile(self):
        for _ in range(WARMUP_STEPS):
            self._run_all_offload()

        TimerStatsStore.clear_stats()
        for _ in range(ACTIVE_STEPS):
            self._run_all_offload()

        time.sleep(10)

        return {
            "time_stats": TimerStatsStore.get_stats(),
            "rank": self._rank,
            "num_workers": self._num_workers,
            "n_embd": self._n_embd,
            "batch_size": self._batch_size,
            "context_length": self._context_length,
        }
