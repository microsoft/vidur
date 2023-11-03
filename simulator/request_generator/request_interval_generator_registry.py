from simulator.request_generator.gamma_request_interval_generator import (
    GammaRequestIntervalGenerator,
)
from simulator.request_generator.poisson_request_interval_generator import (
    PoissonRequestIntervalGenerator,
)
from simulator.request_generator.static_request_interval_generator import (
    StaticRequestIntervalGenerator,
)
from simulator.request_generator.trace_request_interval_generator import (
    TraceRequestIntervalGenerator,
)
from simulator.types import RequestIntervalGeneratorType
from simulator.utils.base_registry import BaseRegistry


class RequestIntervalGeneratorRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> RequestIntervalGeneratorType:
        return RequestIntervalGeneratorType.from_str(key_str)


RequestIntervalGeneratorRegistry.register(
    RequestIntervalGeneratorType.GAMMA, GammaRequestIntervalGenerator
)
RequestIntervalGeneratorRegistry.register(
    RequestIntervalGeneratorType.POISSON, PoissonRequestIntervalGenerator
)
RequestIntervalGeneratorRegistry.register(
    RequestIntervalGeneratorType.STATIC, StaticRequestIntervalGenerator
)
RequestIntervalGeneratorRegistry.register(
    RequestIntervalGeneratorType.TRACE, TraceRequestIntervalGenerator
)
