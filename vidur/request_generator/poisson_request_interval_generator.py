import math
import random

from vidur.request_generator.base_request_interval_generator import (
    BaseRequestIntervalGenerator,
)


class PoissonRequestIntervalGenerator(BaseRequestIntervalGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._qps = self._config.poisson_request_interval_generator_qps
        self._std = 1.0 / self._qps
        self._max_interval = self._std * 3.0

    def get_next_inter_request_time(self) -> float:
        next_interval = -math.log(1.0 - random.random()) / self._qps
        next_interval = min(next_interval, self._max_interval)
        return next_interval
