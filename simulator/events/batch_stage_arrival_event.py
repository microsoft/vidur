from typing import List

from simulator.entities.batch import Batch
from simulator.events import BaseEvent
from simulator.logger import init_logger
from simulator.metrics import MetricsStore
from simulator.scheduler import BaseGlobalScheduler
from simulator.types import EventType

logger = init_logger(__name__)


class BatchStageArrivalEvent(BaseEvent):
    def __init__(self, time: float, replica_id: int, stage_id: int, batch: Batch):
        super().__init__(time, EventType.BATCH_STAGE_ARRIVAL)

        self._replica_id = replica_id
        self._stage_id = stage_id
        self._batch = batch

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from simulator.events.replica_stage_schedule_event import (
            ReplicaStageScheduleEvent,
        )

        scheduler.get_replica_stage_scheduler(
            self._replica_id, self._stage_id
        ).add_batch(self._batch)

        return [
            ReplicaStageScheduleEvent(
                self.time,
                self._replica_id,
                self._stage_id,
            )
        ]

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "replica_id": self._replica_id,
            "stage_id": self._stage_id,
            "batch_id": self._batch.id,
        }
