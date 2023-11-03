import math
import random

from simulator.request_generator.base_request_interval_generator import (
    BaseRequestIntervalGenerator,
)


class PoissonRequestIntervalGenerator(BaseRequestIntervalGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._qps = self._config.poisson_request_interval_generator_qps

    def get_next_inter_request_time(self) -> float:
        return -math.log(1.0 - random.random()) / self._qps
