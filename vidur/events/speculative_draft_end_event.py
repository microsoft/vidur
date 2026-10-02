from typing import List

from vidur.entities import Request
from vidur.events.base_event import BaseEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType

logger = init_logger(__name__)


class SpeculativeDraftEndEvent(BaseEvent):
    """Draft tokens for these requests are ready, so they can be verified."""

    def __init__(
        self,
        time: float,
        replica_id: int,
        requests: List[Request],
        replica_epoch: int,
    ):
        super().__init__(time, EventType.SPECULATIVE_DRAFT_END)

        self._replica_id = replica_id
        self._requests = requests
        self._replica_epoch = replica_epoch

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        return scheduler.failover_manager.on_speculative_draft_end(
            self.time, self._replica_id, self._requests, self._replica_epoch
        )

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "replica_id": self._replica_id,
            "request_ids": [request.id for request in self._requests],
        }
