from abc import ABC, abstractmethod
from typing import List

from simulator.plotting import MetricsStore
from simulator.scheduler import BaseGlobalScheduler


class BaseEvent(ABC):
    _id = 0

    def __init__(self, time: float):
        self._time = time
        self._id = BaseEvent.generate_id()

    @classmethod
    def generate_id(cls):
        cls._id += 1
        return cls._id

    @property
    def id(self) -> int:
        return self._id

    @property
    def time(self):
        return self._time

    @property
    def event_type(self):
        pass

    @abstractmethod
    def handle_event(
        self,
        current_time: float,
        scheduler: BaseGlobalScheduler,
        metrics_store: MetricsStore,
    ) -> List["BaseEvent"]:
        pass

    def __lt__(self, other):
        if self.time == other.time:
            if self.event_type == other.event_type:
                return self.id < other.id
            return self.event_type < other.event_type
        else:
            return self.time < other.time

    def __eq__(self, other):
        return (
            self.time == other.time
            and self.event_type == other.event_type
            and self.id == other.id
        )

    def __str__(self) -> str:
        # use to_dict to get a dict representation of the object
        # and convert it to a string
        class_name = self.__class__.__name__
        return f"{class_name}({str(self.to_dict())})"

    def to_dict(self):
        return {"time": self.time, "event_type": self.event_type}

    def to_chrome_trace(self) -> dict:
        return None
