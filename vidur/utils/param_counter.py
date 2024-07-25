from math import ceil

from vidur.config import SimulationConfig


class ParamCounter:
    def __init__(self, config: SimulationConfig) -> None:
        replica_config = config.cluster_config.replica_config
        model_config = replica_config.model_config

        self._embedding_dim = model_config.embedding_dim
        self._num_pipeline_stages = replica_config.num_pipeline_stages
        self._num_tensor_parallel_workers = replica_config.tensor_parallel_size
        self._num_layers = model_config.num_layers
        self._num_q_heads = model_config.num_q_heads
        self._num_kv_heads = model_config.num_kv_heads
        self._mlp_hidden_dim = model_config.mlp_hidden_dim
        self._use_gated_mlp = model_config.use_gated_mlp
        self._vocab_size = model_config.vocab_size

        assert self._num_q_heads % self._num_tensor_parallel_workers == 0
        assert self._num_layers % self._num_pipeline_stages == 0
        assert self._embedding_dim % self._num_tensor_parallel_workers == 0
        assert self._embedding_dim % self._num_q_heads == 0

        self._num_layers_per_pipeline_stage = (
            self._num_layers // self._num_pipeline_stages
        )
        self._attention_head_dim = self._embedding_dim // self._num_q_heads
        self._q_heads_per_tensor_parallel_worker = (
            self._num_q_heads // self._num_tensor_parallel_workers
        )
        self._kv_heads_per_tensor_parallel_worker = ceil(
            self._num_kv_heads / self._num_tensor_parallel_workers
        )

    def get_num_parameters_per_layer(self) -> int:
        num_parameters = 0
        # weights for attention metrics Wq, Wk, Wv
        num_parameters += (
            self._embedding_dim
            * self._attention_head_dim
            * (
                self._q_heads_per_tensor_parallel_worker
                + 2 * self._kv_heads_per_tensor_parallel_worker
            )
        )
        # weights for attention metrics Wo
        num_parameters += (
            self._embedding_dim
            * self._attention_head_dim
            * self._q_heads_per_tensor_parallel_worker
        )
        # fc layer weights
        if self._use_gated_mlp:
            num_parameters += (
                3
                * self._embedding_dim
                * self._mlp_hidden_dim
                // self._num_tensor_parallel_workers
            )
        else:
            num_parameters += (
                2
                * self._embedding_dim
                * self._mlp_hidden_dim
                // self._num_tensor_parallel_workers
            )

        return num_parameters

    def get_num_parameters_per_device(self) -> int:
        num_parameters_per_layer = self.get_num_parameters_per_layer()
        return num_parameters_per_layer * self._num_layers_per_pipeline_stage
