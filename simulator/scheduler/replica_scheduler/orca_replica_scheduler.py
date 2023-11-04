from simulator.entities.batch import Batch
from simulator.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)


class OrcaReplicaScheduler(BaseReplicaScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._preempted_requests = []
        self._num_running_batches = 0
        self._use_single_prefill_per_batch = (
            self._config.orca_scheduler_use_single_prefill_per_batch
        )

    def on_batch_end(self, batch: Batch) -> None:
        self._num_running_batches -= 1

        for request in batch.requests:
            if request.completed:
                self.free(request.id)
            else:
                self._preempted_requests.append(request)

    def _get_next_batch(self) -> Batch:
        requests = []
        num_tokens = []
        contains_prefill = False

        # all preempted_requests will have prefill completed
        while self._preempted_requests:
            if len(requests) == self._max_batch_size:
                break

            request = self._preempted_requests.pop(0)
            next_num_tokens = self._get_request_next_num_tokens(request)
            requests.append(request)
            num_tokens.append(next_num_tokens)

        while self._request_queue:
            if len(requests) == self._max_batch_size:
                break

            if not self.can_allocate(self._max_blocks_per_sequence):
                break

            if self._use_single_prefill_per_batch and contains_prefill:
                break

            request = self._request_queue.pop(0)

            self.allocate(request.id, self._max_blocks_per_sequence)
            next_num_tokens = self._get_request_next_num_tokens(request)
            requests.append(request)
            num_tokens.append(next_num_tokens)
            contains_prefill = True

        if not requests:
            return

        return Batch(self._replica_id, requests, num_tokens)
