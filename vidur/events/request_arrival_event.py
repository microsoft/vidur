from typing import List

from vidur.entities import Request
from vidur.events.base_event import BaseEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType

logger = init_logger(__name__)


class RequestArrivalEvent(BaseEvent):
    def __init__(self, time: float, request: Request) -> None:
        super().__init__(time, EventType.REQUEST_ARRIVAL)

        self._request = request

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from vidur.events.global_schedule_event import GlobalScheduleEvent

        logger.debug(f"Request: {self._request.id} arrived at {self.time}")
        scheduler.add_request(self._request)
        metrics_store.on_request_arrival(self.time, self._request)
        return [GlobalScheduleEvent(self.time)]

    def to_dict(self) -> dict:
        return {
            "time": self.time,
            "event_type": self.event_type,
            "request": self._request.id,
            "priority": getattr(self._request, "priority", 0),
        }
    
    def to_chrome_trace(self):
        """Emit request arrival with priority to Chrome trace."""
        priority = getattr(self._request, "priority", 0)
        return [{
            "name": f"Request {self._request.id} Arrival (P{priority})",
            "cat": "request_lifecycle",
            "ph": "i",  # Instant event
            "ts": self.time * 1e6,
            "pid": -1,  # Global scope
            "tid": 0,
            "s": "g",
            "args": {
                "request_id": self._request.id,
                "priority": priority,
                "num_prefill_tokens": self._request.num_prefill_tokens,
                "num_decode_tokens": self._request.num_decode_tokens,
            }
        }]
