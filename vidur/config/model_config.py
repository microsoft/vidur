from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from vidur.config.base_poly_config import BasePolyConfig
from vidur.logger import init_logger
from vidur.types import NormType, ActivationType, ModelType

logger = init_logger(__name__)


@dataclass
class BaseModelConfig(BasePolyConfig):
    num_layers: int = field(
        metadata={"help": "The number of layers in the model"},
    )
    num_q_heads: int = field(
        metadata={"help": "The number of query heads in the model"},
    )
    num_kv_heads: int = field(
        metadata={"help": "The number of key-value heads in the model"},
    )
    embedding_dim: int = field(
        metadata={"help": "The embedding dimension of the model"},
    )
    mlp_hidden_dim: int = field(
        metadata={"help": "The hidden dimension of the MLP in the model"},
    )
    max_position_embeddings: int = field(
        metadata={"help": "The maximum position embeddings in the model"},
    )
    use_gated_mlp: bool = field(
        metadata={"help": "Whether to use gated MLP in the model"},
    )
    use_bias: bool = field(
        metadata={"help": "Whether to use bias in the model"},
    )
    use_qkv_bias: bool = field(
        metadata={"help": "Whether to use bias in the QKV in the model"},
    )
    activation: ActivationType = field(
        metadata={"help": "The activation function in the model"},
    )
    norm: NormType = field(
        metadata={"help": "The normalization function in the model"},
    )
    post_attn_norm: bool = field(
        metadata={"help": "Whether to use post-attention normalization in the model"},
    )
    vocab_size: int = field(
        metadata={"help": "The vocabulary size of the model"},
    )
    is_neox_style: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use the Neox style in the model"},
    )
    rope_theta: Optional[int] = field(
        default=None,
        metadata={"help": "The rope theta in the model"},
    )
    rope_scaling: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={"help": "The rope scaling config for the model"},
    )
    partial_rotary_factor: float = field(
        default=1.0,
        metadata={"help": "The partial rotary factor in the model"},
    )
    no_tensor_parallel: bool = field(
        default=False,
        metadata={"help": "Whether to use tensor parallelism in the model"},
    )


@dataclass
class Llama2ModelConfig(BaseModelConfig):
    max_position_embeddings: int = field(
        default=16384,
        metadata={"help": "The maximum position embeddings in the model"},
    )
    use_gated_mlp: bool = field(
        default=True,
        metadata={"help": "Whether to use gated MLP in the model"},
    )
    use_bias: bool = field(
        default=False,
        metadata={"help": "Whether to use bias in the model"},
    )
    use_qkv_bias: bool = field(
        default=False,
        metadata={"help": "Whether to use bias in the QKV in the model"},
    )
    activation: ActivationType = field(
        default=ActivationType.SILU,
        metadata={"help": "The activation function in the model"},
    )
    norm: NormType = field(
        default=NormType.RMS_NORM,
        metadata={"help": "The normalization function in the model"},
    )
    post_attn_norm: bool = field(
        default=True,
        metadata={"help": "Whether to use post-attention normalization in the model"},
    )
    vocab_size: int = field(
        default=32768,
        metadata={"help": "The vocabulary size of the model"},
    )
    is_neox_style: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use the Neox style in the model"},
    )
    rope_theta: Optional[int] = field(
        default=10000.0,
        metadata={"help": "The rope theta in the model"},
    )
    rope_scaling: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={"help": "The rope scaling config for the model"},
    )
    partial_rotary_factor: float = field(
        default=1.0,
        metadata={"help": "The partial rotary factor in the model"},
    )
    no_tensor_parallel: bool = field(
        default=False,
        metadata={"help": "Whether to use tensor parallelism in the model"},
    )


@dataclass
class CodeLlama34BModelConfig(Llama2ModelConfig):
    num_layers: int = field(
        default=48,
        metadata={"help": "The number of layers in the model"},
    )
    num_q_heads: int = field(
        default=64,
        metadata={"help": "The number of query heads in the model"},
    )
    num_kv_heads: int = field(
        default=8,
        metadata={"help": "The number of key-value heads in the model"},
    )
    embedding_dim: int = field(
        default=8192,
        metadata={"help": "The embedding dimension of the model"},
    )
    mlp_hidden_dim: int = field(
        default=22016,
        metadata={"help": "The hidden dimension of the MLP in the model"},
    )

    @staticmethod
    def get_type():
        return ModelType.CODE_LLAMA_34B
    
    @staticmethod
    def get_name():
        return "codellama/CodeLlama-34b-Instruct-hf"


@dataclass
class Llama2_7BModelConfig(Llama2ModelConfig):
    num_layers: int = field(
        default=32,
        metadata={"help": "The number of layers in the model"},
    )
    num_q_heads: int = field(
        default=32,
        metadata={"help": "The number of query heads in the model"},
    )
    num_kv_heads: int = field(
        default=32,
        metadata={"help": "The number of key-value heads in the model"},
    )
    embedding_dim: int = field(
        default=4096,
        metadata={"help": "The embedding dimension of the model"},
    )
    mlp_hidden_dim: int = field(
        default=11008,
        metadata={"help": "The hidden dimension of the MLP in the model"},
    )

    @staticmethod
    def get_type():
        return ModelType.LLAMA_2_7B
    
    @staticmethod
    def get_name():
        return "meta-llama/Llama-2-7b-hf"


@dataclass
class Llama2_70BModelConfig(Llama2ModelConfig):
    num_layers: int = field(
        default=80,
        metadata={"help": "The number of layers in the model"},
    )
    num_q_heads: int = field(
        default=64,
        metadata={"help": "The number of query heads in the model"},
    )
    num_kv_heads: int = field(
        default=8,
        metadata={"help": "The number of key-value heads in the model"},
    )
    embedding_dim: int = field(
        default=8192,
        metadata={"help": "The embedding dimension of the model"},
    )
    mlp_hidden_dim: int = field(
        default=28672,
        metadata={"help": "The hidden dimension of the MLP in the model"},
    )

    @staticmethod
    def get_type():
        return ModelType.LLAMA_2_70B
    
    @staticmethod
    def get_name():
        return "meta-llama/Llama-2-70b-hf"


@dataclass
class Llama3_8BModelConfig(Llama2ModelConfig):
    num_layers: int = field(
        default=32,
        metadata={"help": "The number of layers in the model"},
    )
    num_q_heads: int = field(
        default=32,
        metadata={"help": "The number of query heads in the model"},
    )
    num_kv_heads: int = field(
        default=8,
        metadata={"help": "The number of key-value heads in the model"},
    )
    embedding_dim: int = field(
        default=4096,
        metadata={"help": "The embedding dimension of the model"},
    )
    mlp_hidden_dim: int = field(
        default=14336,
        metadata={"help": "The hidden dimension of the MLP in the model"},
    )
    max_position_embeddings: int = field(
        default=4096,
        metadata={"help": "The maximum position embeddings in the model"},
    )
    rope_theta: Optional[int] = field(
        default=500000.0,
        metadata={"help": "The rope theta in the model"},
    )
    vocab_size: int = field(
        default=128256,
        metadata={"help": "The vocabulary size of the model"},
    )


    @staticmethod
    def get_type():
        return ModelType.LLAMA_3_70B
    
    @staticmethod
    def get_name():
        return "meta-llama/Meta-Llama-3-8b"


@dataclass
class Llama3_70BModelConfig(Llama2ModelConfig):
    num_layers: int = field(
        default=80,
        metadata={"help": "The number of layers in the model"},
    )
    num_q_heads: int = field(
        default=64,
        metadata={"help": "The number of query heads in the model"},
    )
    num_kv_heads: int = field(
        default=8,
        metadata={"help": "The number of key-value heads in the model"},
    )
    embedding_dim: int = field(
        default=8192,
        metadata={"help": "The embedding dimension of the model"},
    )
    mlp_hidden_dim: int = field(
        default=28672,
        metadata={"help": "The hidden dimension of the MLP in the model"},
    )
    max_position_embeddings: int = field(
        default=8192,
        metadata={"help": "The maximum position embeddings in the model"},
    )
    rope_theta: Optional[int] = field(
        default=500000.0,
        metadata={"help": "The rope theta in the model"},
    )
    vocab_size: int = field(
        default=128256,
        metadata={"help": "The vocabulary size of the model"},
    )

    @staticmethod
    def get_type():
        return ModelType.LLAMA_3_70B
    
    @staticmethod
    def get_name():
        return "meta-llama/Meta-Llama-3-70B"

@dataclass
class InternLM2ModelConfig(Llama2ModelConfig):
    max_position_embeddings: int = field(
        default=32768,
        metadata={"help": "The maximum position embeddings in the model"},
    )
    vocab_size: int = field(
        default=92544,
        metadata={"help": "The vocabulary size of the model"},
    )


@dataclass
class InternLM2_20BModelConfig(InternLM2ModelConfig):
    num_layers: int = field(
        default=48,
        metadata={"help": "The number of layers in the model"},
    )
    num_q_heads: int = field(
        default=48,
        metadata={"help": "The number of query heads in the model"},
    )
    num_kv_heads: int = field(
        default=8,
        metadata={"help": "The number of key-value heads in the model"},
    )
    embedding_dim: int = field(
        default=6144,
        metadata={"help": "The embedding dimension of the model"},
    )
    mlp_hidden_dim: int = field(
        default=16384,
        metadata={"help": "The hidden dimension of the MLP in the model"},
    )

    @staticmethod
    def get_type():
        return ModelType.INTERNLM_2_20B
    
    @staticmethod
    def get_name():
        return "internlm/internlm2-20b"


@dataclass
class Phi2ModelConfig(Llama2ModelConfig):
    num_layers: int = field(
        default=32,
        metadata={"help": "The number of layers in the model"},
    )
    num_q_heads: int = field(
        default=32,
        metadata={"help": "The number of query heads in the model"},
    )
    num_kv_heads: int = field(
        default=32,
        metadata={"help": "The number of key-value heads in the model"},
    )
    embedding_dim: int = field(
        default=2560,
        metadata={"help": "The embedding dimension of the model"},
    )
    mlp_hidden_dim: int = field(
        default=10240,
        metadata={"help": "The hidden dimension of the MLP in the model"},
    )
    max_position_embeddings: int = field(
        default=2048,
        metadata={"help": "The maximum position embeddings in the model"},
    )
    use_gated_mlp: bool = field(
        default=False,
        metadata={"help": "Whether to use gated MLP in the model"},
    )
    use_bias: bool = field(
        default=True,
        metadata={"help": "Whether to use bias in the model"},
    )
    use_qkv_bias: bool = field(
        default=True,
        metadata={"help": "Whether to use bias in the QKV in the model"},
    )
    activation: ActivationType = field(
        default=ActivationType.GELU,
        metadata={"help": "The activation function in the model"},
    )
    norm: NormType = field(
        default=NormType.LAYER_NORM,
        metadata={"help": "The normalization function in the model"},
    )
    post_attn_norm: bool = field(
        default=False,
        metadata={"help": "Whether to use post-attention normalization in the model"},
    )
    vocab_size: int = field(
        default=51200,
        metadata={"help": "The vocabulary size of the model"},
    )
    rope_scaling: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={"help": "The rope scaling config for the model"},
    )
    rope_theta: Optional[int] = field(
        default=10000.0,
        metadata={"help": "The rope theta in the model"},
    )
    partial_rotary_factor: float = field(
        default=0.4,
        metadata={"help": "The partial rotary factor in the model"},
    )
    no_tensor_parallel: bool = field(
        default=True,
        metadata={"help": "Whether to use tensor parallelism in the model"},
    )
    is_neox_style: bool = field(
        default=True,
        metadata={"help": "Whether to use the Neox style in the model"},
    )

    @staticmethod
    def get_type():
        return ModelType.PHI2
    
    @staticmethod
    def get_name():
        return "microsoft/phi-2"


@dataclass
class QwenModelConfig(Llama2ModelConfig):
    use_qkv_bias: bool = field(
        default=True,
        metadata={"help": "Whether to use bias in the QKV in the model"},
    )
    max_position_embeddings: int = field(
        default=32768,
        metadata={"help": "The maximum position embeddings in the model"},
    )
    vocab_size: int = field(
        default=152064,
        metadata={"help": "The vocabulary size of the model"},
    )


@dataclass
class Qwen72BModelConfig(QwenModelConfig):
    num_layers: int = field(
        default=80,
        metadata={"help": "The number of layers in the model"},
    )
    num_q_heads: int = field(
        default=64,
        metadata={"help": "The number of query heads in the model"},
    )
    num_kv_heads: int = field(
        default=64,
        metadata={"help": "The number of key-value heads in the model"},
    )
    embedding_dim: int = field(
        default=8192,
        metadata={"help": "The embedding dimension of the model"},
    )
    mlp_hidden_dim: int = field(
        default=24576,
        metadata={"help": "The hidden dimension of the MLP in the model"},
    )

    @staticmethod
    def get_type():
        return ModelType.QWEN_72B
    
    @staticmethod
    def get_name():
        return "Qwen/Qwen-72B"
