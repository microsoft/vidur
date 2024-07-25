from dataclasses import dataclass, field

from vidur.config.base_poly_config import BasePolyConfig
from vidur.logger import init_logger
from vidur.types import DeviceSKUType

logger = init_logger(__name__)


@dataclass
class BaseDeviceSKUConfig(BasePolyConfig):
    fp16_tflops: int = field(
        metadata={"help": "The number of TFLOPS the device can achieve in FP16"},
    )
    total_memory_gb: int = field(
        metadata={"help": "The total memory of the device in GB"},
    )


@dataclass
class A100DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = field(
        default=312,
        metadata={"help": "The number of TFLOPS the device can achieve in FP16"},
    )
    total_memory_gb: int = field(
        default=80,
        metadata={"help": "The total memory of the device in GB"},
    )

    @staticmethod
    def get_type():
        return DeviceSKUType.A40


@dataclass
class A40DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = field(
        default=150,
        metadata={"help": "The number of TFLOPS the device can achieve in FP16"},
    )
    total_memory_gb: int = field(
        default=45,
        metadata={"help": "The total memory of the device in GB"},
    )

    @staticmethod
    def get_type():
        return DeviceSKUType.A100


@dataclass
class H100DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = field(
        default=1000,
        metadata={"help": "The number of TFLOPS the device can achieve in FP16"},
    )
    total_memory_gb: int = field(
        default=80,
        metadata={"help": "The total memory of the device in GB"},
    )

    @staticmethod
    def get_type():
        return DeviceSKUType.H100

