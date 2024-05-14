from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from vidur.config import Config
from vidur.entities import Replica, Request
from vidur.execution_time_predictor import ExecutionTimePredictorRegistry
from vidur.scheduler.replica_scheduler.replica_scheduler_registry import (
    ReplicaSchedulerRegistry,
)


class BaseGlobalScheduler(ABC):
    def __init__(self, config: Config, replicas: Dict[int, Replica]):
        self._config = config
        self._replicas = replicas

        self._num_replicas = len(self._replicas)

        execution_time_predictor = ExecutionTimePredictorRegistry.get_from_str(
            self._config.execution_time_predictor_provider,
            self._config,
        )
        self._replica_schedulers = {
            replica_id: ReplicaSchedulerRegistry.get_from_str(
                config.replica_scheduler_provider,
                config,
                replica,
                replica.num_pipeline_stages,
                execution_time_predictor,
            )
            for replica_id, replica in replicas.items()
        }
        self._request_queue = []

    def sort_requests(self) -> None:
        self._request_queue.sort(key=lambda request: request._arrived_at)

    def add_request(self, request: Request) -> None:
        self._request_queue.append(request)

    def get_replica_scheduler(self, replica_id: int):
        return self._replica_schedulers[replica_id]

    def get_replica_stage_scheduler(self, replica_id: int, stage_id: int):
        return self._replica_schedulers[replica_id].get_replica_stage_scheduler(
            stage_id
        )

    def is_empty(self) -> bool:
        return len(self._request_queue) == 0 and all(
            replica_scheduler.is_empty()
            for replica_scheduler in self._replica_schedulers.values()
        )

    @abstractmethod
    def schedule(self) -> List[Tuple[int, Request]]:
        pass
