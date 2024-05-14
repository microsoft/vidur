from typing import List

from simulator.entities import Request
from simulator.events.base_event import BaseEvent
from simulator.logger import init_logger
from simulator.metrics import MetricsStore
from simulator.scheduler import BaseGlobalScheduler
from simulator.types import EventType

logger = init_logger(__name__)


class RequestArrivalEvent(BaseEvent):
    def __init__(self, time: float, request: Request) -> None:
        super().__init__(time, EventType.REQUEST_ARRIVAL)

        self._request = request

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from simulator.events.global_schedule_event import GlobalScheduleEvent

        logger.debug(f"Request: {self._request.id} arrived at {self.time}")
        scheduler.add_request(self._request)
        metrics_store.on_request_arrival(self.time, self._request)
        return [GlobalScheduleEvent(self.time)]

    def to_dict(self) -> dict:
        return {
            "time": self.time,
            "event_type": self.event_type,
            "request": self._request.id,
        }
