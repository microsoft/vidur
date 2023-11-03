from math import ceil

from simulator.entities.batch import Batch, Request
from simulator.scheduler.replica_scheduler.orca_replica_scheduler import (
    OrcaReplicaScheduler,
)


class VLLMReplicaScheduler(OrcaReplicaScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._watermark_blocks_fraction = (
            self._config.vllm_scheduler_watermark_blocks_fraction
        )
        self._max_tokens_in_batch = self._config.vllm_scheduler_max_tokens_in_batch
        # increase the max batch size to allow for more requests to be processed
        self._max_batch_size *= (
            self._config.vllm_scheduler_max_batch_size_amplification_factor
        )
        self._max_batch_size = min(
            self._max_batch_size, self._config.replica_scheduler_batch_size_cap
        )
        self._watermark_blocks = int(
            self._watermark_blocks_fraction * self._num_total_blocks
        )

        assert (
            self._num_stages == 1
        ), "vLLM scheduler does not support pipeline parallel execution."

    def _can_allocate_request(self, request: Request) -> bool:
        if request.id not in self._allocation_map:
            # new request
            num_required_blocks = ceil(
                (request.num_prefill_tokens + 1) / self._block_size
            )
            return (
                self._num_total_blocks
                - self._num_allocated_blocks
                - num_required_blocks
                >= self._watermark_blocks
            )

        # vllm requires at least one block to be available
        return self._num_total_blocks - self._num_allocated_blocks >= 1

    def _allocate_request(self, request: Request) -> None:
        if request.id not in self._allocation_map:
            # new request
            num_required_blocks = ceil(
                (request.num_prefill_tokens + 1) / self._block_size
            )
            self.allocate(request.id, num_required_blocks)
            return

        num_tokens_reserved = self._allocation_map[request.id] * self._block_size
        num_tokens_required = max(
            0, request.num_processed_tokens + 1 - num_tokens_reserved
        )
        assert (
            num_tokens_required == 0 or num_tokens_required == 1
        ), f"num_tokens_required: {num_tokens_required}"

        if num_tokens_required == 0:
            return

        self.allocate(request.id, 1)

    def _get_next_batch(self) -> Batch:
        requests = []
        num_tokens = []
        num_batch_tokens = 0

        while self._request_queue:
            request = self._request_queue[0]

            next_num_tokens = self._get_request_next_num_tokens(request)

            if not self._can_allocate_request(request):
                break

            if num_batch_tokens + next_num_tokens > self._max_tokens_in_batch:
                break

            if len(self._preempted_requests) + len(requests) + 1 > self._max_batch_size:
                break

            request = self._request_queue.pop(0)

            self._allocate_request(request)
            requests.append(request)
            num_tokens.append(next_num_tokens)
            num_batch_tokens += next_num_tokens

        if requests:
            return Batch(self._replica_id, requests, num_tokens)

        # all preempted_requests will have prefill completed
        while self._preempted_requests:
            if len(requests) + 1 > self._max_batch_size:
                break

            request = self._preempted_requests.pop(0)

            while not self._can_allocate_request(request):
                if self._preempted_requests:
                    victim_request = self._preempted_requests.pop(-1)
                    victim_request.restart()
                    self.free(victim_request.id)
                    self._request_queue = [victim_request] + self._request_queue
                else:
                    request.restart()
                    self.free(request.id)
                    self._request_queue = [request] + self._request_queue
                    break
            else:
                self._allocate_request(request)
                next_num_tokens = self._get_request_next_num_tokens(request)
                requests.append(request)
                num_tokens.append(next_num_tokens)

        if not requests:
            return

        return Batch(self._replica_id, requests, num_tokens)
