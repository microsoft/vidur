from random import randint
from typing import List, Tuple

from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler


class RandomGlobalScheduler(BaseGlobalScheduler):
    def schedule(self) -> List[Tuple[int, Request]]:
        self.sort_requests()

        request_mapping = []
        replica_ids = self.routable_replica_ids
        while self._request_queue and replica_ids:
            request = self._request_queue.pop(0)
            replica_id = replica_ids[randint(1, len(replica_ids)) - 1]
            request_mapping.append((replica_id, request))
        return request_mapping
