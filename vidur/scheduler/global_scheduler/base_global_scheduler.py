from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from vidur.config import SimulationConfig
from vidur.entities import Replica, Request
from vidur.execution_time_predictor import ExecutionTimePredictorRegistry
from vidur.logger import init_logger
from vidur.scheduler.replica_scheduler.replica_scheduler_registry import (
    ReplicaSchedulerRegistry,
)


logger = init_logger(__name__)

class BaseGlobalScheduler(ABC):
    def __init__(self, config: SimulationConfig, replicas: Dict[int, Replica]):
        self._config = config
        self._replicas = replicas
        self._num_replicas = len(self._replicas)

        # --------------------------------------------------------
        # Select execution time predictor
        # --------------------------------------------------------
        execution_time_predictor = ExecutionTimePredictorRegistry.get(
            config.execution_time_predictor_config.get_type(),
            predictor_config=config.execution_time_predictor_config,
            replica_config=config.cluster_config.replica_config,
            replica_scheduler_config=config.cluster_config.replica_scheduler_config,
            metrics_config=config.metrics_config,
        )

        # --------------------------------------------------------
        # Determine which replica scheduler type to use
        # --------------------------------------------------------
        global_type = str(config.cluster_config.global_scheduler_config.get_type()).lower()
        configured_replica_type = config.cluster_config.replica_scheduler_config.get_type()

        if global_type == "llumnix":
            # Check if user explicitly configured a non-llumlet replica scheduler
            if configured_replica_type != "llumlet":
                logger.warning(
                    f"Llumnix global scheduler requested with '{configured_replica_type}' replica scheduler. "
                    f"Note: Llumnix features (migration, priority headroom, virtual usage) require 'llumlet'. "
                    f"Using '{configured_replica_type}' - some Llumnix features may be unavailable."
                )
                replica_sched_type = configured_replica_type
            else:
                logger.info("Global scheduler is Llumnix → using LlumletLocalScheduler per replica (paper-compliant).")
                replica_sched_type = "llumlet"
        else:
            # Use the normal one from config
            replica_sched_type = configured_replica_type
            logger.info(f"Using replica scheduler type: {replica_sched_type}")

        # --------------------------------------------------------
        # Construct one scheduler per replica
        # --------------------------------------------------------
        self._replica_schedulers = {
            replica_id: ReplicaSchedulerRegistry.get(
                replica_sched_type,
                replica_config=config.cluster_config.replica_config,
                replica_scheduler_config=config.cluster_config.replica_scheduler_config,
                request_generator_config=config.request_generator_config,
                replica=replica,
                num_stages=replica.num_pipeline_stages,
                execution_time_predictor=execution_time_predictor,
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
