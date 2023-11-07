from abc import ABC, abstractmethod
#from typing import List

from simulator.config import Config
from simulator.entities import Batch


class BaseExecutionTimePredictor(ABC):
    def __init__(self, config: Config) -> None:
        self._num_tensor_parallel_workers = config.replica_num_tensor_parallel_workers
        self._num_pipeline_stages = config.replica_num_pipeline_stages
        self._num_layers_per_pipeline_stage = (
            config.replica_num_layers // config.replica_num_pipeline_stages
        )

    def get_execution_time(self, batch: Batch, pipeline_stage: int) -> float:
        # we are not counting the execution time for the embedding layer and last softmax layer
        layer_execution_time = self._get_layer_execution_time(batch)
        pipeline_stage_execution_time = (
            layer_execution_time * self._num_layers_per_pipeline_stage
        )
        if pipeline_stage == self._num_pipeline_stages - 1:
            pipeline_parallel_communication_time = 0
        else:
            pipeline_parallel_communication_time = (
                self._get_pipeline_parallel_communication_time(batch)
            )
        # return in seconds
        return (
            pipeline_stage_execution_time + pipeline_parallel_communication_time
        ) * 1e-3

    def _get_layer_execution_time(self, batch: Batch) -> float:
        attention_layer_time = self._get_attention_layer_execution_time(batch)
        mlp_layer_time = self._get_mlp_layer_execution_time(batch)

        return attention_layer_time + mlp_layer_time

    def _get_attention_layer_execution_time(self, batch: Batch) -> float:
        pre_proj_time = self._get_attention_layer_pre_proj_execution_time(batch)
        post_proj_time = self._get_attention_layer_post_proj_execution_time(batch)
        flash_attention_time = self._get_attention_layer_flash_attention_execution_time(
            batch
        )
        if self._num_tensor_parallel_workers == 1:
            tensor_parallel_communication_time = 0
        else:
            tensor_parallel_communication_time = (
                self._get_tensor_parallel_communication_time(batch)
            )
        return (
            pre_proj_time
            + post_proj_time
            + flash_attention_time
            + tensor_parallel_communication_time
        )

    def _get_mlp_layer_execution_time(self, batch: Batch) -> float:
        mlp_time = self._get_mlp_layer_mlp_execution_time(batch)
        if self._num_tensor_parallel_workers == 1:
            tensor_parallel_communication_time = 0
        else:
            tensor_parallel_communication_time = (
                self._get_tensor_parallel_communication_time(batch)
            )
        return mlp_time + tensor_parallel_communication_time

    @abstractmethod
    def _get_attention_layer_pre_proj_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_attention_layer_post_proj_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_attention_layer_flash_attention_execution_time(
        self, batch: Batch
    ) -> float:
        pass

    @abstractmethod
    def _get_mlp_layer_mlp_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_tensor_parallel_communication_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_pipeline_parallel_communication_time(self, batch: Batch) -> float:
        pass
