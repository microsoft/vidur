import torch

from benchmark.timer_stats_store import TimerStatsStore


class CudaTimer:
    def __init__(self, name, aggregation_fn=sum, filter_str=None):
        self.name = name
        self.aggregation_fn = aggregation_fn
        self.filter_str = filter_str

        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=self.handle_trace,
        )

    def __enter__(self):
        self.profiler.__enter__()
        return self

    def handle_trace(self, trace):
        events = trace.events()

        if self.filter_str:
            events = [e for e in events if self.filter_str in e.name]

        total_cuda_time = self.aggregation_fn([e.cuda_time_total for e in trace.events()])
        TimerStatsStore.record_time(self.name, total_cuda_time * 1e-3) # convert to ms

    def __exit__(self, *args):
        self.profiler.__exit__(*args)
