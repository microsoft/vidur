import logging
from math import ceil

from simulator.config import Config
from simulator.entities.base_entity import BaseEntity

logger = logging.getLogger(__name__)


class Replica(BaseEntity):
    def __init__(self, config: Config) -> None:
        assert config.replica_num_layers % config.replica_num_pipeline_stages == 0
        assert (
            config.replica_embedding_dim % config.replica_num_tensor_parallel_workers
            == 0
        )

        self._id = Replica.generate_id()

        self._num_pipeline_stages = config.replica_num_pipeline_stages
        self._num_tensor_parallel_workers = config.replica_num_tensor_parallel_workers
        self._num_layers = config.replica_num_layers
        self._num_q_heads = config.replica_num_q_heads
        self._num_kv_heads = config.replica_num_kv_heads
        self._embedding_dim = config.replica_embedding_dim
        self._mlp_hidden_dim = config.replica_mlp_hidden_dim
        self._use_gated_mlp = config.replica_use_gated_mlp
        self._vocab_size = config.replica_vocab_size
        self._total_memory_gb = config.replica_total_memory_gb
        self._memory_margin_fraction = config.replica_memory_margin_fraction
        self._max_request_tokens = config.request_generator_max_tokens
        self._per_device_flops = config.replica_fp16_tflops * 2**40

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_q_heads(self) -> int:
        return self._num_q_heads

    @property
    def num_kv_heads(self) -> int:
        return self._num_kv_heads

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def mlp_hidden_dim(self) -> int:
        return self._mlp_hidden_dim

    @property
    def use_gated_mlp(self) -> int:
        return self._use_gated_mlp

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def num_pipeline_stages(self) -> int:
        return self._num_pipeline_stages

    @property
    def num_layers_per_pipeline_stage(self) -> int:
        return self._num_layers // self._num_pipeline_stages

    @property
    def attention_head_dim(self) -> int:
        return self._embedding_dim // self._num_q_heads

    @property
    def q_heads_per_tensor_parallel_worker(self) -> int:
        return self._num_q_heads // self._num_tensor_parallel_workers

    @property
    def kv_heads_per_tensor_parallel_worker(self) -> int:
        return ceil(self._num_kv_heads / self._num_tensor_parallel_workers)

    @property
    def num_tensor_parallel_workers(self) -> int:
        return self._num_tensor_parallel_workers

    @property
    def total_memory_gb(self) -> int:
        return self._total_memory_gb

    @property
    def memory_margin_fraction(self) -> float:
        return self._memory_margin_fraction

    @property
    def max_request_tokens(self) -> int:
        return self._max_request_tokens

    @property
    def per_device_flops(self) -> float:
        return self._per_device_flops

    def to_dict(self) -> dict:
        return {
            "id": self._id,
            "num_layers": self._num_layers,
            "num_q_heads": self._num_q_heads,
            "num_kv_heads": self._num_kv_heads,
            "embedding_dim": self._embedding_dim,
            "mlp_hidden_dim": self._mlp_hidden_dim,
            "use_gated_mlp": self._use_gated_mlp,
            "vocab_size": self._vocab_size,
            "num_pipeline_stages": self._num_pipeline_stages,
            "num_tensor_parallel_workers": self._num_tensor_parallel_workers,
        }
