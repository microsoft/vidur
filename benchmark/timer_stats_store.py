import numpy as np


class TimerStatsStore:
    TIMING_STATS = {}

    @classmethod
    def record_time(cls, name, time):
        if name not in cls.TIMING_STATS:
            cls.TIMING_STATS[name] = []

        cls.TIMING_STATS[name].append(time)

    @classmethod
    def clear_stats(cls):
        cls.TIMING_STATS = {}

    @classmethod
    def get_stats(cls):
        stats = {}
        for name, times in cls.TIMING_STATS.items():
            stats[name] = {
                "min": np.min(times),
                "max": np.max(times),
                "mean": np.mean(times),
                "median": np.median(times),
                "std": np.std(times),
            }

        return stats
