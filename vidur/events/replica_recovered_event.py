from typing import List

from vidur.events.base_event import BaseEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType

logger = init_logger(__name__)


class ReplicaRecoveredEvent(BaseEvent):
    def __init__(self, time: float, replica_id: int):
        super().__init__(time, EventType.REPLICA_RECOVERED)

        self._replica_id = replica_id

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        return scheduler.failover_manager.on_replica_recovered(
            self.time, self._replica_id
        )

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "replica_id": self._replica_id,
        }

    def to_chrome_trace(self) -> dict:
        return {
            "name": "recovered",
            "ph": "i",
            "s": "p",
            "ts": self.time * 1e6,
            "pid": self._replica_id,
            "tid": 0,
        }
