from simulator.config import Config
from simulator.entities import BatchStage
from simulator.utils.param_counter import ParamCounter


class MFUCalculator:
    def __init__(self, config: Config):
        param_counter = ParamCounter(config)
        self._num_params_per_device = param_counter.get_num_parameters_per_device()
        self._num_layers_per_device = (
            config.replica_num_layers // config.replica_num_pipeline_stages
        )
        self._embedding_dim = config.replica_embedding_dim
        self._num_heads_per_device = (
            config.replica_num_q_heads // config.replica_num_tensor_parallel_workers
        )
        self._head_dimension = self._embedding_dim // config.replica_num_q_heads
        self._device_flops = config.replica_fp16_tflops * 2**40

    def _get_mlp_flops(self, batch_stage: BatchStage) -> float:
        num_tokens = sum(batch_stage.num_tokens)
        return 2 * num_tokens * self._num_params_per_device

    def _get_attention_flops(self, batch_stage: BatchStage) -> float:
        total_flops = 0
        for request, num_tokens in zip(batch_stage.requests, batch_stage.num_tokens):
            total_flops += (
                4  # for number of ops in attention
                * self._num_layers_per_device
                * self._num_heads_per_device
                * self._head_dimension
                * num_tokens  # q length
                * (num_tokens + request.num_processed_tokens)  # kv length
            )

        return total_flops

    def get_mfu(self, batch_stage: BatchStage) -> float:
        mlp_flops = self._get_mlp_flops(batch_stage)
        attention_flops = self._get_attention_flops(batch_stage)
        total_flops = mlp_flops + attention_flops
        total_flops_per_second = total_flops / batch_stage.execution_time
        return total_flops_per_second * 100 / self._device_flops
