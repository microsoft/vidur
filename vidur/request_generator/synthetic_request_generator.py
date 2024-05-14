from typing import List

from vidur.entities import Request
from vidur.request_generator.base_request_generator import BaseRequestGenerator
from vidur.request_generator.request_interval_generator_registry import (
    RequestIntervalGeneratorRegistry,
)
from vidur.request_generator.request_length_generator_registry import (
    RequestLengthGeneratorRegistry,
)
from vidur.utils.random import set_seeds


class SyntheticRequestGenerator(BaseRequestGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._seed = self._config.seed

        self._request_length_generator = RequestLengthGeneratorRegistry.get_from_str(
            self._config.synthetic_request_generator_length_provider, self._config
        )
        self._request_interval_generator = (
            RequestIntervalGeneratorRegistry.get_from_str(
                self._config.synthetic_request_generator_interval_provider, self._config
            )
        )

    def _generate_next_request(self, last_arrived_at: float) -> Request:
        inter_request_time = (
            self._request_interval_generator.get_next_inter_request_time()
        )
        if inter_request_time is None:
            return None
        arrived_at = last_arrived_at + inter_request_time

        (
            prefill_tokens,
            decode_tokens,
        ) = self._request_length_generator.get_next_num_tokens()

        if prefill_tokens is None or decode_tokens is None:
            return None

        return Request(
            arrived_at=arrived_at,
            num_prefill_tokens=int(prefill_tokens),
            num_decode_tokens=int(decode_tokens),
        )

    def _generate_requests(self) -> List[Request]:
        requests = []

        current_time = 0

        # first priority is duration
        if self._config.synthetic_request_generator_duration is not None:
            while current_time < self._config.synthetic_request_generator_duration:
                request = self._generate_next_request(current_time)
                current_time = request.arrived_at
                requests.append(request)
        elif self._config.synthetic_request_generator_num_requests is not None:
            for _ in range(self._config.synthetic_request_generator_num_requests):
                request = self._generate_next_request(current_time)
                current_time = request.arrived_at
                requests.append(request)
        else:
            assert self._config.synthetic_request_generator_interval_provider == "trace"
            while True:
                request = self._generate_next_request(current_time)
                if request is None:
                    break
                current_time = request.arrived_at
                requests.append(request)

        return requests

    def generate_requests(self) -> List[Request]:
        assert (
            self._config.synthetic_request_generator_num_requests
            or self._config.synthetic_request_generator_duration
            or self._config.synthetic_request_generator_interval_provider == "trace"
        )

        set_seeds(self._seed)

        requests = self._generate_requests()

        # sort requests by arrival time
        requests.sort(key=lambda x: (x.arrived_at, x.id))
        # remove any requests that arrived after the time limit
        if self._config.synthetic_request_generator_duration is not None:
            requests = [
                request
                for request in requests
                if request.arrived_at
                < self._config.synthetic_request_generator_duration
            ]

        return requests
