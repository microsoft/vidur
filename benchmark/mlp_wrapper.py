import time
import os

import torch

from benchmark.nanogpt import (
    Block as NanoGPTBlock,
    GPTConfig as NanoGPTConfig,
)
from benchmark.timer_stats_store import TimerStatsStore

WARMUP_STEPS = 2
ACTIVE_STEPS = 5


class ModelWrapper:
    def __init__(
        self,
        model_config,
        num_tokens,
        num_tensor_parallel_workers=1,
        device="cuda"
    ):
        super().__init__()

        self.n_head = model_config[0]
        self.n_kv_head = model_config[1]
        self.n_embd = model_config[2]
        self.n_expanded_embd = model_config[3]
        self.vocab_size = model_config[4]
        self.use_gated_mlp = model_config[5]
        self.device = device
        self.num_tensor_parallel_workers = num_tensor_parallel_workers
        self.num_tokens = num_tokens

        os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
        # add torch performance benchmarking flags
        torch.backends.cudnn.benchmark = True

        config = NanoGPTConfig(
            n_head=self.n_head,
            n_kv_head=self.n_kv_head,
            n_embd=self.n_embd,
            n_expanded_embd=self.n_expanded_embd,
            use_gated_mlp=self.use_gated_mlp,
            vocab_size=self.vocab_size,
            dropout=0,
            bias=False,
            num_tensor_parallel_workers=num_tensor_parallel_workers,
        )
        self.raw_model = NanoGPTBlock(config)
        self.raw_model = self.raw_model.to(dtype=torch.float16)
        # self.model = torch.compile(self.raw_model)
        self.model = self.raw_model

    def forward(self, x):
        try:
            torch.cuda.synchronize()
            self.model.forward(x)
            torch.cuda.synchronize()
        finally:
            # do nothing
            pass

    def profile(self):
        fits_in_memory = True

        try:
            x = None
            # If OOM occurs during the profiling, there is no way to recover
            # from it. This is probably a bug in PyTorch. So we need to
            # run one step without profiling to make sure we don't OOM during profiling.
            self.model.eval()
            self.model.to(self.device)

            vocab_range = self.vocab_size // self.num_tensor_parallel_workers
            x = torch.randint(
                low=0, high=vocab_range, size=(self.num_tokens,), device=self.device, dtype=torch.int32
            )

            for _ in range(WARMUP_STEPS):
                self.forward(x)

            TimerStatsStore.clear_stats()

            for _ in range(ACTIVE_STEPS):
                self.forward(x)

        except RuntimeError as e:
            raise e
            print(
                f"Skipping num_tokens {self.num_tokens} due to error: {e}",
                flush=True,
            )
            fits_in_memory = False
        finally:
            del self.model, self.raw_model, x
            # wait till deletes are logged
            time.sleep(1)
            # force free memory buffers to avoid pt2 memory leaks
            # torch.cuda.empty_cache()

        if fits_in_memory:
            stats = {
                "time_stats": TimerStatsStore.get_stats(),
                "n_head": self.n_head,
                "n_kv_head": self.n_kv_head,
                "n_embd": self.n_embd,
                "n_expanded_embd": self.n_expanded_embd,
                "vocab_size": self.vocab_size,
                "use_gated_mlp": self.use_gated_mlp,
                "num_tokens": self.num_tokens,
                "num_tensor_parallel_workers": self.num_tensor_parallel_workers,
            }
            TimerStatsStore.clear_stats()
        else:
            stats = {}

        return fits_in_memory, stats
