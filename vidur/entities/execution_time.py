from vidur.entities.base_entity import BaseEntity


class ExecutionTime(BaseEntity):
    def __init__(
        self,
        num_layers_per_pipeline_stage: int,
        attention_rope_execution_time: float,
        attention_kv_cache_save_execution_time: float,
        attention_decode_execution_time: float,
        attention_prefill_execution_time: float,
        attention_layer_pre_proj_execution_time: float,
        attention_layer_post_proj_execution_time: float,
        mlp_layer_up_proj_execution_time: float,
        mlp_layer_down_proj_execution_time: float,
        mlp_layer_act_execution_time: float,
        attn_norm_time: float,
        mlp_norm_time: float,
        add_time: float,
        tensor_parallel_communication_time: float,
        pipeline_parallel_communication_time: float,
        schedule_time: float,
        sampler_e2e_time: float,
        prepare_inputs_e2e_time: float,
        process_model_outputs_time: float,
        ray_comm_time: float,
    ) -> None:
        self._id = ExecutionTime.generate_id()

        self._num_layers_per_pipeline_stage = num_layers_per_pipeline_stage
        self._attention_rope_execution_time = attention_rope_execution_time
        self._attention_kv_cache_save_execution_time = (
            attention_kv_cache_save_execution_time
        )
        self._attention_decode_execution_time = attention_decode_execution_time
        self._attention_prefill_execution_time = attention_prefill_execution_time
        self._attention_layer_pre_proj_execution_time = (
            attention_layer_pre_proj_execution_time
        )
        self._attention_layer_post_proj_execution_time = (
            attention_layer_post_proj_execution_time
        )
        self._mlp_layer_up_proj_execution_time = mlp_layer_up_proj_execution_time
        self._mlp_layer_down_proj_execution_time = mlp_layer_down_proj_execution_time
        self._mlp_layer_act_execution_time = mlp_layer_act_execution_time
        self._mlp_norm_time = mlp_norm_time
        self._attn_norm_time = attn_norm_time
        self._add_time = add_time
        self._tensor_parallel_communication_time = tensor_parallel_communication_time
        self._pipeline_parallel_communication_time = (
            pipeline_parallel_communication_time
        )
        self._schedule_time = schedule_time
        self._sampler_e2e_time = sampler_e2e_time
        self._prepare_inputs_e2e_time = prepare_inputs_e2e_time
        self._process_model_outputs_time = process_model_outputs_time
        self._ray_comm_time = ray_comm_time

    def _get_mlp_layer_execution_time(self) -> float:
        return (
            self._mlp_layer_up_proj_execution_time
            + self._mlp_layer_down_proj_execution_time
            + self._mlp_layer_act_execution_time
            + self._tensor_parallel_communication_time
            + self._mlp_norm_time
        )

    def _get_attention_layer_execution_time(self) -> float:
        return (
            self._attention_layer_pre_proj_execution_time
            + self._attention_layer_post_proj_execution_time
            + self._attention_rope_execution_time
            + self._attention_kv_cache_save_execution_time
            + self._attention_decode_execution_time
            + self._attention_prefill_execution_time
            + self._tensor_parallel_communication_time
            + self._attn_norm_time
        )

    def _get_block_execution_time(self) -> float:
        return (
            self._get_attention_layer_execution_time()
            + self._get_mlp_layer_execution_time()
            + self._add_time
        )

    def _get_cpu_overhead(self) -> float:
        return (
            self._schedule_time
            + self._sampler_e2e_time
            + self._prepare_inputs_e2e_time
            + self._process_model_outputs_time
            + self._ray_comm_time
        )

    @property
    def num_layers(self) -> int:
        return self._num_layers_per_pipeline_stage

    @property
    def mlp_layer_up_proj_execution_time(self) -> float:
        return self._mlp_layer_up_proj_execution_time

    @property
    def mlp_layer_down_proj_execution_time(self) -> float:
        return self._mlp_layer_down_proj_execution_time

    @property
    def mlp_layer_act_execution_time(self) -> float:
        return self._mlp_layer_act_execution_time

    @property
    def mlp_all_reduce_time(self) -> float:
        return self._tensor_parallel_communication_time

    @property
    def attention_pre_proj_time(self) -> float:
        return self._attention_layer_pre_proj_execution_time

    @property
    def attention_post_proj_time(self) -> float:
        return self._attention_layer_post_proj_execution_time

    @property
    def attention_all_reduce_time(self) -> float:
        return self._tensor_parallel_communication_time

    @property
    def attention_rope_execution_time(self) -> float:
        return self._attention_rope_execution_time

    @property
    def attention_kv_cache_save_execution_time(self) -> float:
        return self._attention_kv_cache_save_execution_time

    @property
    def attention_decode_execution_time(self) -> float:
        return self._attention_decode_execution_time

    @property
    def attention_prefill_execution_time(self) -> float:
        return self._attention_prefill_execution_time

    @property
    def pipeline_parallel_communication_time(self) -> float:
        return self._pipeline_parallel_communication_time

    @property
    def schedule_time(self) -> float:
        return self._schedule_time

    @property
    def sampler_e2e_time(self) -> float:
        return self._sampler_e2e_time

    @property
    def prepare_inputs_e2e_time(self) -> float:
        return self._prepare_inputs_e2e_time

    @property
    def process_model_outputs_time(self) -> float:
        return self._process_model_outputs_time

    @property
    def ray_comm_time(self) -> float:
        return self._ray_comm_time

    @property
    def mlp_norm_time(self) -> float:
        return self._mlp_norm_time

    @property
    def attn_norm_time(self) -> float:
        return self._attn_norm_time

    @property
    def add_time(self) -> float:
        return self._add_time

    @property
    def model_time(self) -> float:
        # we are not counting the execution time for the embedding layer and last softmax layer
        block_execution_time = self._get_block_execution_time()
        pipeline_stage_execution_time = (
            block_execution_time * self._num_layers_per_pipeline_stage
        )
        # return in seconds
        return (
            pipeline_stage_execution_time + self.pipeline_parallel_communication_time
        ) * 1e-3

    @property
    def model_time_ms(self) -> float:
        return self.model_time * 1e3

    @property
    def total_time(self) -> float:
        # return in seconds
        return self.model_time + self._get_cpu_overhead() * 1e-3
