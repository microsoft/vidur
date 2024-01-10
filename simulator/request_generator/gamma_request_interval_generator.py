from scipy.stats import gamma

from simulator.request_generator.base_request_interval_generator import (
    BaseRequestIntervalGenerator,
)


class GammaRequestIntervalGenerator(BaseRequestIntervalGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        cv = self._config.gamma_request_interval_generator_cv
        self._qps = self._config.gamma_request_interval_generator_qps
        self._gamma_shape = 1.0 / (cv**2)

    def get_next_inter_request_time(self) -> float:
        gamma_scale = 1.0 / (self._qps * self._gamma_shape)
        return gamma.rvs(self._gamma_shape, scale=gamma_scale)
