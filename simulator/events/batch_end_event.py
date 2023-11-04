import logging
from typing import List

from simulator.entities import Batch
from simulator.events import BaseEvent
from simulator.plotting import MetricsStore
from simulator.scheduler import BaseGlobalScheduler
from simulator.types import EventType

logger = logging.getLogger(__name__)


class BatchEndEvent(BaseEvent):
    def __init__(self, time: float, replica_id: int, batch: Batch):
        super().__init__(time)
        self._replica_id = replica_id
        self._batch = batch

    @property
    def event_type(self):
        return EventType.BATCH_END

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from simulator.events.replica_schedule_event import ReplicaScheduleEvent

        self._batch.on_batch_end(self.time)
        replica_scheduler = scheduler.get_replica_scheduler(self._replica_id)
        replica_scheduler.on_batch_end(self._batch)

        memory_usage_percent = replica_scheduler.memory_usage_percent
        metrics_store.on_batch_end(
            self.time, self._batch, self._replica_id, memory_usage_percent
        )

        return [ReplicaScheduleEvent(self.time, self._replica_id)]

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "batch_id": self._batch.id,
        }
