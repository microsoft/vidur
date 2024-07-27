from dataclasses import dataclass, field

from vidur.config.base_fixed_config import BaseFixedConfig
from vidur.logger import init_logger
from vidur.types import NodeSKUType, DeviceSKUType

logger = init_logger(__name__)


@dataclass
class BaseNodeSKUConfig(BaseFixedConfig):
    num_devices_per_node: int = field(
        metadata={"help": "The number of devices per node"},
    )


@dataclass
class A40PairwiseNvlinkNodeSKUConfig(BaseNodeSKUConfig):
    device_sku_type: DeviceSKUType = field(
        default=DeviceSKUType.A40,
        metadata={"help": "The device SKU type"},
    )
    num_devices_per_node: int = field(
        default=8,
        metadata={"help": "The number of devices per node"},
    )

    @staticmethod
    def get_type():
        return NodeSKUType.A40_PAIRWISE_NVLINK


@dataclass
class A100PairwiseNvlinkNodeSKUConfig(BaseNodeSKUConfig):
    device_sku_type: DeviceSKUType = field(
        default=DeviceSKUType.A100,
        metadata={"help": "The device SKU type"},
    )
    num_devices_per_node: int = field(
        default=8,
        metadata={"help": "The number of devices per node"},
    )

    @staticmethod
    def get_type():
        return NodeSKUType.A100_PAIRWISE_NVLINK


@dataclass
class H100PairwiseNvlinkNodeSKUConfig(BaseNodeSKUConfig):
    device_sku_type: DeviceSKUType = field(
        default=DeviceSKUType.H100,
        metadata={"help": "The device SKU type"},
    )
    num_devices_per_node: int = field(
        default=8,
        metadata={"help": "The number of devices per node"},
    )

    @staticmethod
    def get_type():
        return NodeSKUType.H100_PAIRWISE_NVLINK


@dataclass
class A100DgxNodeSKUConfig(BaseNodeSKUConfig):
    device_sku_type: DeviceSKUType = field(
        default=DeviceSKUType.A100,
        metadata={"help": "The device SKU type"},
    )
    num_devices_per_node: int = field(
        default=8,
        metadata={"help": "The number of devices per node"},
    )

    @staticmethod
    def get_type():
        return NodeSKUType.A100_DGX


@dataclass
class H100DgxNodeSKUConfig(BaseNodeSKUConfig):
    device_sku_type: DeviceSKUType = field(
        default=DeviceSKUType.H100,
        metadata={"help": "The device SKU type"},
    )
    num_devices_per_node: int = field(
        default=8,
        metadata={"help": "The number of devices per node"},
    )

    @staticmethod
    def get_type():
        return NodeSKUType.H100_DGX
