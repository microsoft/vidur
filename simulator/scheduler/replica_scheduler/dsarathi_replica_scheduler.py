from math import ceil

from simulator.entities.batch import Batch, Request
from simulator.scheduler.replica_scheduler.sarathi_replica_scheduler import (
    SarathiReplicaScheduler,
)


class DSarathiReplicaScheduler(SarathiReplicaScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # sarathi config
        self._chunk_size = self._config.dsarathi_scheduler_chunk_size
        # club multiple prefills to ensure uniform chunk size
        self._enable_rolling_prefills = (
            self._config.dsarathi_scheduler_enable_rolling_prefills
        )
        # when we are packing multiple prefills in a batch, we need to ensure
        # that we don't end up packing a very small prefill chunk just to make batch full
        # because that will lead to reduced number of schedulable prefill requests
        self._prefill_fitting_tolerance = (
            self._config.dsarathi_scheduler_prefill_fitting_tolerance
        )
        # vLLM config
        self._watermark_blocks_fraction = (
            self._config.dsarathi_scheduler_watermark_blocks_fraction
        )
        # increase the max batch size to allow for more requests to be processed
        self._max_batch_size *= (
            self._config.dsarathi_scheduler_max_batch_size_amplification_factor
        )
        self._max_batch_size = min(
            self._max_batch_size, self._config.replica_scheduler_batch_size_cap
        )
        self._watermark_blocks = int(
            self._watermark_blocks_fraction * self._num_total_blocks
        )

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
        skipped_requests = []
        contains_prefill = False
        num_batch_tokens = 0

        # preempted requests could contain multiple requests which have
        # partial prefills completed, so we need to be careful
        while self._preempted_requests:
            if len(requests) == self._max_batch_size:
                break

            request = self._preempted_requests.pop(0)
            next_num_tokens = self._get_request_next_num_tokens(
                request, contains_prefill, num_batch_tokens
            )

            if next_num_tokens == 0:
                skipped_requests.append(request)
                continue

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

                if not request.is_prefill_complete:
                    contains_prefill = True

                num_batch_tokens += next_num_tokens
                requests.append(request)
                num_tokens.append(next_num_tokens)

        # re-add the skipped requests, but make sure that we add them to the
        # front of the queue so that they are scheduled first and we maintain FIFO ordering
        self._preempted_requests = skipped_requests + self._preempted_requests
        skipped_requests = []

        while self._request_queue:
            if len(requests) == self._max_batch_size:
                break

            if not self._can_allocate_request(self._request_queue[0]):
                break

            next_num_tokens = self._get_request_next_num_tokens(
                self._request_queue[0], contains_prefill, num_batch_tokens
            )

            if next_num_tokens == 0:
                break

            request = self._request_queue.pop(0)

            self._allocate_request(request)

            # all new requests will have a prefill
            contains_prefill = True
            num_batch_tokens += next_num_tokens
            requests.append(request)
            num_tokens.append(next_num_tokens)

        if not requests:
            return

        return Batch(self._replica_id, requests, num_tokens)
