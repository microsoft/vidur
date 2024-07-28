from math import ceil

from vidur.config import BaseRequestGeneratorConfig, ReplicaConfig
from vidur.entities.base_entity import BaseEntity
from vidur.logger import init_logger

logger = init_logger(__name__)


class Replica(BaseEntity):
    def __init__(
        self,
        replica_config: ReplicaConfig,
        generator_config: BaseRequestGeneratorConfig,
    ) -> None:
        self._id = Replica.generate_id()

        self._replica_config = replica_config
        self._model_config = replica_config.model_config
        self._device_config = replica_config.device_config
        self._generator_config = generator_config

        assert (
            self._model_config.num_layers % self._replica_config.num_pipeline_stages
            == 0
        )
        assert (
            self._model_config.embedding_dim % self._replica_config.tensor_parallel_size
            == 0
        )

    @property
    def id(self) -> int:
        return self._id

    @property
    def num_layers(self) -> int:
        return self._model_config.num_layers

    @property
    def num_q_heads(self) -> int:
        return self._model_config.num_q_heads

    @property
    def num_kv_heads(self) -> int:
        return self._model_config.num_kv_heads

    @property
    def embedding_dim(self) -> int:
        return self._model_config.embedding_dim

    @property
    def mlp_hidden_dim(self) -> int:
        return self._model_config.mlp_hidden_dim

    @property
    def use_gated_mlp(self) -> int:
        return self._model_config.use_gated_mlp

    @property
    def vocab_size(self) -> int:
        return self._model_config.vocab_size

    @property
    def num_pipeline_stages(self) -> int:
        return self._replica_config.num_pipeline_stages

    @property
    def num_layers_per_pipeline_stage(self) -> int:
        return self._model_config.num_layers // self._replica_config.num_pipeline_stages

    @property
    def attention_head_dim(self) -> int:
        return self._model_config.embedding_dim // self._model_config.num_q_heads

    @property
    def q_heads_per_tensor_parallel_worker(self) -> int:
        return (
            self._model_config.num_q_heads // self._replica_config.tensor_parallel_size
        )

    @property
    def kv_heads_per_tensor_parallel_worker(self) -> int:
        return ceil(
            self._model_config.num_kv_heads / self._replica_config.tensor_parallel_size
        )

    @property
    def num_tensor_parallel_workers(self) -> int:
        return self._replica_config.tensor_parallel_size

    @property
    def total_memory_gb(self) -> int:
        return self._device_config.total_memory_gb

    @property
    def memory_margin_fraction(self) -> float:
        return self._replica_config.memory_margin_fraction

    @property
    def max_request_tokens(self) -> int:
        return self._generator_config.max_tokens

    @property
    def per_device_flops(self) -> float:
        return self._device_config.fp16_tflops * 2**40

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "num_layers": self.num_layers,
            "num_q_heads": self.num_q_heads,
            "num_kv_heads": self.num_kv_heads,
            "embedding_dim": self.embedding_dim,
            "mlp_hidden_dim": self.mlp_hidden_dim,
            "use_gated_mlp": self.use_gated_mlp,
            "vocab_size": self.vocab_size,
            "num_pipeline_stages": self.num_pipeline_stages,
            "num_tensor_parallel_workers": self.num_tensor_parallel_workers,
        }
