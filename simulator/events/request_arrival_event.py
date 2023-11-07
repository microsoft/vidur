import logging
from typing import List

from simulator.entities import Request
from simulator.events.base_event import BaseEvent
from simulator.plotting import MetricsStore
from simulator.scheduler import BaseGlobalScheduler
from simulator.types import EventType
from simulator.events.global_schedule_event import GlobalScheduleEvent

logger = logging.getLogger(__name__)


class RequestArrivalEvent(BaseEvent):
    def __init__(self, time: float, request: Request) -> None:
        super().__init__(time)
        self._request = request

    @property
    def event_type(self) -> EventType:
        return EventType.REQUEST_ARRIVAL

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:

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
