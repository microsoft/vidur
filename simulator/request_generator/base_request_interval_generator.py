from abc import ABC, abstractmethod

from simulator.config import Config


class BaseRequestIntervalGenerator(ABC):
    def __init__(self, config: Config):
        self._config = config

    @abstractmethod
    def get_next_inter_request_time(self) -> float:
        pass
