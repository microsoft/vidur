from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from vidur.config.base_fixed_config import BaseFixedConfig
from vidur.logger import init_logger
from vidur.types import ActivationType, NormType

logger = init_logger(__name__)


@dataclass
class BaseModelConfig(BaseFixedConfig):
    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    embedding_dim: int
    mlp_hidden_dim: int
    max_position_embeddings: int
    use_gated_mlp: bool
    use_bias: bool
    use_qkv_bias: bool
    activation: ActivationType
    norm: NormType
    post_attn_norm: bool
    vocab_size: int
    is_neox_style: Optional[bool] = True
    rope_theta: Optional[float] = None
    rope_scaling: Optional[Dict[str, Any]] = None
    partial_rotary_factor: float = 1.0
    no_tensor_parallel: bool = False


@dataclass
class Llama2ModelConfig(BaseModelConfig):
    max_position_embeddings: int = 16384
    use_gated_mlp: bool = True
    use_bias: bool = False
    use_qkv_bias: bool = False
    activation: ActivationType = ActivationType.SILU
    norm: NormType = NormType.RMS_NORM
    post_attn_norm: bool = True
    vocab_size: int = 32768
    is_neox_style: Optional[bool] = True
    rope_theta: Optional[float] = 10000
    rope_scaling: Optional[Dict[str, Any]] = None
    partial_rotary_factor: float = 1.0
    no_tensor_parallel: bool = False

    @staticmethod
    def get_name():
        return "meta-llama/Llama-2-Config"


@dataclass
class CodeLlama34BModelConfig(Llama2ModelConfig):
    num_layers: int = 48
    num_q_heads: int = 64
    num_kv_heads: int = 8
    embedding_dim: int = 8192
    mlp_hidden_dim: int = 22016
    rope_theta: Optional[float] = 1000000

    @staticmethod
    def get_name():
        return "codellama/CodeLlama-34b-Instruct-hf"


@dataclass
class Llama2_7BModelConfig(Llama2ModelConfig):
    num_layers: int = 32
    num_q_heads: int = 32
    num_kv_heads: int = 32
    embedding_dim: int = 4096
    mlp_hidden_dim: int = 11008
    max_position_embeddings: int = 4096

    @staticmethod
    def get_name():
        return "meta-llama/Llama-2-7b-hf"


@dataclass
class Llama2_70BModelConfig(Llama2ModelConfig):
    num_layers: int = 80
    num_q_heads: int = 64
    num_kv_heads: int = 8
    embedding_dim: int = 8192
    mlp_hidden_dim: int = 28672
    max_position_embeddings: int = 4096

    @staticmethod
    def get_name():
        return "meta-llama/Llama-2-70b-hf"


@dataclass
class Llama3_8BModelConfig(Llama2ModelConfig):
    num_layers: int = 32
    num_q_heads: int = 32
    num_kv_heads: int = 8
    embedding_dim: int = 4096
    mlp_hidden_dim: int = 14336
    max_position_embeddings: int = 4096
    rope_theta: Optional[float] = 500000
    vocab_size: int = 128256

    @staticmethod
    def get_name():
        return "meta-llama/Meta-Llama-3-8B"


@dataclass
class Llama3_70BModelConfig(Llama2ModelConfig):
    num_layers: int = 80
    num_q_heads: int = 64
    num_kv_heads: int = 8
    embedding_dim: int = 8192
    mlp_hidden_dim: int = 28672
    max_position_embeddings: int = 8192
    rope_theta: Optional[float] = 500000
    vocab_size: int = 128256

    @staticmethod
    def get_name():
        return "meta-llama/Meta-Llama-3-70B"


@dataclass
class InternLMModelConfig(Llama2ModelConfig):
    max_position_embeddings: int = 4096
    vocab_size: int = 103168


@dataclass
class InternLM_20BModelConfig(InternLMModelConfig):
    num_layers: int = 60
    num_q_heads: int = 40
    num_kv_heads: int = 40
    embedding_dim: int = 5120
    mlp_hidden_dim: int = 13824

    @staticmethod
    def get_name():
        return "internlm/internlm-20b"


@dataclass
class InternLM2ModelConfig(Llama2ModelConfig):
    max_position_embeddings: int = 32768
    vocab_size: int = 92544


@dataclass
class InternLM2_20BModelConfig(InternLM2ModelConfig):
    num_layers: int = 48
    num_q_heads: int = 48
    num_kv_heads: int = 8
    embedding_dim: int = 6144
    mlp_hidden_dim: int = 16384
    rope_theta: Optional[float] = 1000000

    @staticmethod
    def get_name():
        return "internlm/internlm2-20b"


@dataclass
class Phi2ModelConfig(Llama2ModelConfig):
    num_layers: int = 32
    num_q_heads: int = 32
    num_kv_heads: int = 32
    embedding_dim: int = 2560
    mlp_hidden_dim: int = 10240
    max_position_embeddings: int = 2048
    use_gated_mlp: bool = False
    use_bias: bool = True
    use_qkv_bias: bool = True
    activation: ActivationType = ActivationType.GELU
    norm: NormType = NormType.LAYER_NORM
    post_attn_norm: bool = False
    vocab_size: int = 51200
    rope_scaling: Optional[Dict[str, Any]] = None
    rope_theta: Optional[float] = 10000
    partial_rotary_factor: float = 0.4
    no_tensor_parallel: bool = True

    @staticmethod
    def get_name():
        return "microsoft/phi-2"


@dataclass
class QwenModelConfig(Llama2ModelConfig):
    use_qkv_bias: bool = True
    max_position_embeddings: int = 32768
    vocab_size: int = 152064

    @staticmethod
    def get_name():
        return "Qwen/Qwen-Config"


@dataclass
class Qwen72BModelConfig(QwenModelConfig):
    num_layers: int = 80
    num_q_heads: int = 64
    num_kv_heads: int = 64
    embedding_dim: int = 8192
    mlp_hidden_dim: int = 24576
    rope_theta: Optional[float] = 1000000

    @staticmethod
    def get_name():
        return "Qwen/Qwen-72B"


# ──────────────────────────────────────────────────────────────────────────────
# Mixture-of-Experts (MoE) model configs
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class BaseMoEModelConfig(BaseModelConfig):
    """Base class for Mixture-of-Experts models.


    Extends ``BaseModelConfig`` with the fields required to simulate
    MoE-specific execution: expert count, top-k routing, and the expert
    intermediate dimension (which differs from the dense MLP hidden dim).

    These fields feed the ``MoELayerExecutionTimePredictor`` (see
    ``vidur/execution_time_predictor/moe_execution_time_predictor.py``),
    which models expert load imbalance and routing overhead separately
    from the attention path.
    """

    # MoE topology
    num_experts: int = 0
    num_active_experts: int = 0  # top-k per token
    expert_intermediate_dim: int = 0  # per-expert FFN hidden dim
    num_shared_experts: int = 0  # DeepSeek-style always-active experts

    # Multi-head Latent Attention (MLA) — used by DeepSeek models
    kv_lora_rank: int = 0  # 0 means standard MHA, >0 means MLA
    q_lora_rank: Optional[int] = None

    # Expert parallelism degree (for simulation sweep)
    expert_parallel_degree: int = 1

    @staticmethod
    def get_name():
        return None  # Abstract base — subclasses must override


@dataclass
class DeepSeekV3ModelConfig(BaseMoEModelConfig):
    """Model configuration for DeepSeek-V3 (671B total, 37B active per token).

    Architecture reference: DeepSeek-V3 Technical Report (2024).
    - 61 transformer layers
    - Multi-head Latent Attention (MLA): kv_lora_rank=512, q_lora_rank=1536
    - 256 routed experts + 1 shared expert per MoE layer
    - Top-8 routing (8 experts activated per token)
    - Expert intermediate dim: 2048 per expert
    - Dense layers 0–2 use standard MLP (not MoE)

    Profiling on RTX 3070Ti for latency prediction targets A100/H100 via
    hardware ratio scaling (documented in profiling_data/deepseek_v3/).
    """

    # Attention
    num_layers: int = 61
    num_q_heads: int = 128
    num_kv_heads: int = 128
    embedding_dim: int = 7168
    mlp_hidden_dim: int = 18432  # dense MLP used in first 3 layers
    max_position_embeddings: int = 131072
    use_gated_mlp: bool = True
    use_bias: bool = False
    use_qkv_bias: bool = False
    activation: ActivationType = ActivationType.SILU
    norm: NormType = NormType.RMS_NORM
    post_attn_norm: bool = True
    vocab_size: int = 129280
    rope_theta: Optional[float] = 10000.0
    is_neox_style: Optional[bool] = True

    # MoE
    num_experts: int = 256
    num_active_experts: int = 8
    expert_intermediate_dim: int = 2048
    num_shared_experts: int = 1

    # MLA
    kv_lora_rank: int = 512
    q_lora_rank: Optional[int] = 1536

    @staticmethod
    def get_name():
        return "deepseek-ai/DeepSeek-V3"


@dataclass
class MixtralModelConfig(BaseMoEModelConfig):
    """Model configuration for Mixtral 8x7B (Mistral AI).

    Architecture reference: Mixtral of Experts (Jiang et al., 2024).
    - Standard sparse MoE: 8 experts, top-2 routing per token
    - No MLA (standard grouped-query attention)
    """

    num_layers: int = 32
    num_q_heads: int = 32
    num_kv_heads: int = 8
    embedding_dim: int = 4096
    mlp_hidden_dim: int = 14336  # per-expert FFN hidden dim
    max_position_embeddings: int = 32768
    use_gated_mlp: bool = True
    use_bias: bool = False
    use_qkv_bias: bool = False
    activation: ActivationType = ActivationType.SILU
    norm: NormType = NormType.RMS_NORM
    post_attn_norm: bool = False
    vocab_size: int = 32000
    rope_theta: Optional[float] = 1000000.0
    is_neox_style: Optional[bool] = True

    # MoE
    num_experts: int = 8
    num_active_experts: int = 2
    expert_intermediate_dim: int = 14336
    num_shared_experts: int = 0

    @staticmethod
    def get_name():
        return "mistralai/Mixtral-8x7B-v0.1"
