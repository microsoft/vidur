from vidur.types import NodeSKUType

NETWORK_DEVICE_MAPPING = {
    "a40_pair_nvlink": NodeSKUType.A40_PAIRWISE_NVLINK,
    "a100_pair_nvlink": NodeSKUType.A100_PAIRWISE_NVLINK,
    "h100_pair_nvlink": NodeSKUType.H100_PAIRWISE_NVLINK,
    "a100_dgx": NodeSKUType.A100_DGX,
    "h100_dgx": NodeSKUType.H100_DGX,
}
