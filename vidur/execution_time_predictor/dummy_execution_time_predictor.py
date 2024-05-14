import random
from typing import List

from vidur.entities import Request
from vidur.execution_time_predictor.base_execution_time_predictor import (
    BaseExecutionTimePredictor,
)


class DummyExecutionTimePredictor(BaseExecutionTimePredictor):
    def _get_attention_layer_pre_proj_execution_time(
        self, batch: List[Request]
    ) -> float:
        return random.uniform(0.1, 0.2)

    def _get_attention_layer_post_proj_execution_time(
        self, batch: List[Request]
    ) -> float:
        return random.uniform(0.1, 0.2)

    def _get_attention_layer_flash_attention_execution_time(
        self, batch: List[Request]
    ) -> float:
        return random.uniform(0.1, 0.2)

    def _get_mlp_layer_mlp_execution_time(self, batch: List[Request]) -> float:
        return random.uniform(0.1, 0.2)

    def _get_tensor_parallel_communication_time(self, batch: List[Request]) -> float:
        return random.uniform(0.1, 0.2)

    def _get_pipeline_parallel_communication_time(self, batch: List[Request]) -> float:
        return random.uniform(0.1, 0.2)
