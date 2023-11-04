import logging
from typing import List

from simulator.events import BaseEvent
from simulator.plotting import MetricsStore
from simulator.scheduler import BaseGlobalScheduler
from simulator.types import EventType

logger = logging.getLogger(__name__)


class ReplicaStageScheduleEvent(BaseEvent):
    def __init__(self, time: float, replica_id: int, stage_id: int):
        super().__init__(time)

        self._replica_id = replica_id
        self._stage_id = stage_id

        self._batch = None
        self._batch_stage = None
        self._is_last_stage = None

    @property
    def event_type(self):
        return EventType.REPLICA_STAGE_SCHEDULE

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from simulator.events.batch_stage_end_event import BatchStageEndEvent

        stage_scheduler = scheduler.get_replica_stage_scheduler(
            self._replica_id, self._stage_id
        )
        self._batch, self._batch_stage = stage_scheduler.on_schedule()

        if not (self._batch and self._batch_stage):
            return []

        self._batch_stage.on_schedule(self.time)
        metrics_store.on_replica_stage_schedule(
            self.time, self._replica_id, self._stage_id, self._batch_stage
        )

        self._is_last_stage = stage_scheduler.is_last_stage

        return [
            BatchStageEndEvent(
                self.time + self._batch_stage.execution_time,
                self._replica_id,
                self._stage_id,
                self._is_last_stage,
                self._batch,
                self._batch_stage,
            ),
        ]

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "replica_id": self._replica_id,
            "stage_id": self._stage_id,
            "batch_id": self._batch.id if self._batch else None,
            "batch_stage_id": self._batch_stage.id if self._batch_stage else None,
            "is_last_stage": self._is_last_stage,
        }
