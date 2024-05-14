from abc import ABC, abstractmethod
from typing import Tuple

from vidur.config import Config


class BaseRequestLengthGenerator(ABC):
    def __init__(self, config: Config):
        self._config = config

    @abstractmethod
    def get_next_num_tokens(self) -> Tuple[float, float]:
        pass
