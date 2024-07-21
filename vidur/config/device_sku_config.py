from dataclasses import dataclass

from vidur.config.base_poly_config import BasePolyConfig
from vidur.logger import init_logger
from vidur.types import DeviceSKUType

logger = init_logger(__name__)


@dataclass
class BaseDeviceSKUConfig(BasePolyConfig):
    fp16_tflops: int
    total_memory_gb: int
    num_devices_per_node: int


@dataclass
class A100DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 312
    total_memory_gb: int = 80

    @staticmethod
    def get_type():
        return DeviceSKUType.A100


@dataclass
class A40DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 150
    total_memory_gb: int = 45

    @staticmethod
    def get_type():
        return DeviceSKUType.A100


@dataclass
class H100DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 1000
    total_memory_gb: int = 80

    @staticmethod
    def get_type():
        return DeviceSKUType.A100

