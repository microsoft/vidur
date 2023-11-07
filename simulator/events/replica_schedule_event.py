import logging
from typing import List

from simulator.events import BaseEvent
from simulator.plotting import MetricsStore
from simulator.scheduler import BaseGlobalScheduler
from simulator.types import EventType
from simulator.events.batch_stage_arrival_event import BatchStageArrivalEvent

logger = logging.getLogger(__name__)


class ReplicaScheduleEvent(BaseEvent):
    def __init__(self, time: float, replica_id: int):
        super().__init__(time)

        self._replica_id = replica_id

        self._batches = []

    @property
    def event_type(self):
        return EventType.REPLICA_SCHEDULE

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:

        replica_scheduler = scheduler.get_replica_scheduler(self._replica_id)
        self._batches = replica_scheduler.on_schedule()

        if not self._batches:
            return []

        memory_usage_percent = replica_scheduler.memory_usage_percent
        metrics_store.on_replica_schedule(
            self.time, self._replica_id, memory_usage_percent
        )

        for batch in self._batches:
            batch.on_schedule(self.time)

        return [
            BatchStageArrivalEvent(
                self.time,
                self._replica_id,
                0,  # stage_id
                batch,
            )
            for batch in self._batches
        ]

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "replica_id": self._replica_id,
            "batch_ids": [batch.id for batch in self._batches],
        }
