from typing import List

from vidur.entities import Request
from vidur.events.base_event import BaseEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType

logger = init_logger(__name__)


class RequestRestoreEndEvent(BaseEvent):
    """An interrupted request is ready on its recovery replica: its KV checkpoint
    (if any) has been copied from host memory to the GPUs."""

    def __init__(
        self,
        time: float,
        replica_id: int,
        request: Request,
        replica_epoch: int,
        num_reserved_tokens: int,
    ):
        super().__init__(time, EventType.REQUEST_RESTORE_END)

        self._replica_id = replica_id
        self._request = request
        self._replica_epoch = replica_epoch
        self._num_reserved_tokens = num_reserved_tokens

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        return scheduler.failover_manager.on_request_restore_end(
            self.time,
            self._replica_id,
            self._request,
            self._replica_epoch,
            self._num_reserved_tokens,
        )

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "replica_id": self._replica_id,
            "request": self._request.id,
        }
