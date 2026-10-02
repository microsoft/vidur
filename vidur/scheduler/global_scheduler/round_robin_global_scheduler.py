from typing import List, Tuple

from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler


class RoundRobinGlobalScheduler(BaseGlobalScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._request_counter = 0

    def schedule(self) -> List[Tuple[int, Request]]:
        self.sort_requests()

        request_mapping = []
        while self._request_queue and self._routable_replica_ids:
            request = self._request_queue.pop(0)
            replica_id = self._request_counter % self._num_replicas
            while replica_id not in self._routable_replica_ids:
                self._request_counter += 1
                replica_id = self._request_counter % self._num_replicas
            self._request_counter += 1
            request_mapping.append((replica_id, request))

        return request_mapping
