from time import perf_counter


class CPUEvent:
    def __init__(self):
        self._time = None

    def record(self):
        self._time = perf_counter()

    @property
    def time(self):
        assert self._time is not None
        return self._time

    def elapsed_time(self, other):
        return other.time - self._time
