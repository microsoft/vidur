from math import ceil

from simulator.config import Config


class ParamCounter:
    def __init__(self, config: Config) -> None:
        self._embedding_dim = config.replica_embedding_dim
        self._num_pipeline_stages = config.replica_num_pipeline_stages
        self._num_tensor_parallel_workers = config.replica_num_tensor_parallel_workers
        self._num_layers = config.replica_num_layers
        self._num_q_heads = config.replica_num_q_heads
        self._num_kv_heads = config.replica_num_kv_heads
        self._embedding_dim = config.replica_embedding_dim
        self._mlp_hidden_dim = config.replica_mlp_hidden_dim
        self._use_gated_mlp = config.replica_use_gated_mlp
        self._vocab_size = config.replica_vocab_size

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
