import numpy as np

from simulator.profiling.utils.singleton import Singleton
from simulator.profiling.utils import ProfileMethod


class TimerStatsStore(metaclass=Singleton):
    def __init__(self, profile_method: ProfileMethod, disabled: bool = False):
        self.disabled = disabled
        self._profile_method = profile_method
        self.TIMING_STATS = {}

    @property
    def profile_method(self):
        return self._profile_method

    def record_time(self, name: str, time):
        name = name.replace("vidur_", "")
        if name not in self.TIMING_STATS:
            self.TIMING_STATS[name] = []

        self.TIMING_STATS[name].append(time)

    def clear_stats(self):
        self.TIMING_STATS = {}

    def get_stats(self):
        stats = {}
        for name, times in self.TIMING_STATS.items():
            times = [
                (time if isinstance(time, float) else time[0].elapsed_time(time[1]))
                for time in times
            ]

            stats[name] = {
                "min": np.min(times),
                "max": np.max(times),
                "mean": np.mean(times),
                "median": np.median(times),
                "std": np.std(times),
            }

        return stats
