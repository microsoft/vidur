from simulator.request_generator.trace_request_length_generator import (
    TraceRequestLengthGenerator,
)
from simulator.request_generator.uniform_request_length_generator import (
    UniformRequestLengthGenerator,
)
from simulator.request_generator.zipf_request_length_generator import (
    ZipfRequestLengthGenerator,
)
from simulator.types import RequestLengthGeneratorType
from simulator.utils.base_registry import BaseRegistry


class RequestLengthGeneratorRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> RequestLengthGeneratorType:
        return RequestLengthGeneratorType.from_str(key_str)


RequestLengthGeneratorRegistry.register(
    RequestLengthGeneratorType.ZIPF, ZipfRequestLengthGenerator
)
RequestLengthGeneratorRegistry.register(
    RequestLengthGeneratorType.UNIFORM, UniformRequestLengthGenerator
)
RequestLengthGeneratorRegistry.register(
    RequestLengthGeneratorType.TRACE, TraceRequestLengthGenerator
)
