
from benchmark.cpu_event import CPUEvent
from benchmark.timer_stats_store import TimerStatsStore


class CPUTimer:
    def __init__(self, name):
        self.name = f"cpu_{name}"
        self.start_event = CPUEvent()
        self.end_event = CPUEvent()

    def __enter__(self):
        self.start_event.record()
        return self

    def __exit__(self, *_):
        self.end_event.record()
        TimerStatsStore.record_stats(self.name, (self.start_event, self.end_event))
