from vidur.types import ModelType, NodeSKUType

MODEL_NAME_MAPPING = {
    "Qwen/Qwen-72B": ModelType.QWEN_72B,
    "codellama/CodeLlama-34b-Instruct-hf": ModelType.CODE_LLAMA_34B,
    "internlm/internlm2-20b": ModelType.INTERNLM_2_20B,
    "meta-llama/Llama-2-7b-hf": ModelType.LLAMA_2_7B,
    "meta-llama/Llama-2-70b-hf": ModelType.LLAMA_2_70B,
    "meta-llama/Meta-Llama-3-8b": ModelType.LLAMA_3_8B,
    "meta-llama/Meta-Llama-3-70B": ModelType.LLAMA_3_70B,
    "microsoft/phi-2": ModelType.PHI2,
}

NETWORK_DEVICE_MAPPING = {
    "a40_pair_nvlink": NodeSKUType.A40_PAIRWISE_NVLINK,
    "a100_pair_nvlink": NodeSKUType.A100_PAIRWISE_NVLINK,
    "h100_pair_nvlink": NodeSKUType.H100_PAIRWISE_NVLINK,
    "a100_dgx": NodeSKUType.A100_DGX,
    "h100_dgx": NodeSKUType.H100_DGX,
}
