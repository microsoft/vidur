import random
from typing import Tuple

from simulator.request_generator.base_request_length_generator import (
    BaseRequestLengthGenerator,
)
from simulator.utils.zipf_generator import ZipfGenerator


class ZipfRequestLengthGenerator(BaseRequestLengthGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._zipf_generator = ZipfGenerator(
            self._config.synthetic_request_generator_min_tokens,
            self._config.request_generator_max_tokens,
            self._config.zipf_request_length_generator_theta,
            self._config.zipf_request_length_generator_scramble,
            self._config.seed,
        )

    def get_next_num_tokens(self) -> Tuple[float, float]:
        total_tokens = self._zipf_generator.next()

        decode_tokens = total_tokens / (
            1 + self._config.synthetic_request_generator_prefill_to_decode_ratio
        )
        prefill_tokens = total_tokens - decode_tokens

        return prefill_tokens, decode_tokens
