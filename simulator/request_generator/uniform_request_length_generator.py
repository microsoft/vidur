import random
from typing import Tuple

from simulator.request_generator.base_request_length_generator import (
    BaseRequestLengthGenerator,
)


class UniformRequestLengthGenerator(BaseRequestLengthGenerator):
    def get_next_num_tokens(self) -> Tuple[float, float]:
        total_tokens = random.uniform(
            self._config.synthetic_request_generator_min_tokens,
            self._config.request_generator_max_tokens,
        )

        decode_tokens = total_tokens / (
            1 + self._config.synthetic_request_generator_prefill_to_decode_ratio
        )
        prefill_tokens = total_tokens - decode_tokens

        return prefill_tokens, decode_tokens
