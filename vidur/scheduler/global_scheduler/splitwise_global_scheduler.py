from typing import List, Tuple, Dict

from vidur.config import Config
from vidur.entities import Replica, Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler


class SplitwiseGlobalScheduler(BaseGlobalScheduler):
    """
    Splitwise global scheduler.
    """
    def __init__(self, config: Config, replicas: Dict[int, Replica]):
        self._config = config
        self._replicas = replicas
        self._num_replicas = len(self._replicas)

        self._pd_node_ratio = self._config.splitwise_scheduler_pd_node_ratio
        self._sub_scheduler = self._config.splitwise_scheduler_sub_scheduler

        assert self._sub_scheduler != "splitwise"

        self._num_prefill_nodes = self._num_replicas * self._pd_node_ratio
        self._num_decode_nodes = self._num_replicas - self._num_prefill_nodes

        assert self._num_prefill_nodes > 0
        assert self._num_decode_nodes > 0

        self._prefill_replicas = {
            replica_id: replica
            for replica_id, replica in self._replicas.items()
            if replica_id < self._num_prefill_nodes
        }

        self._decode_replicas = {
            replica_id: replica
            for replica_id, replica in self._replicas.items()
            if replica_id >= self._num_prefill_nodes
        }

        self._prefill_scheduler = self.get_global_scheduler(self._prefill_replicas)
        self._decode_scheduler = self.get_global_scheduler(self._decode_replicas)

    def get_global_scheduler(self, replicas: Dict[int, Replica]):
        from vidur.scheduler.global_scheduler.global_scheduler_registry import GlobalSchedulerRegistry
        return GlobalSchedulerRegistry.get_from_str(self._sub_scheduler, self._config, replicas)

    def add_request(self, request: Request) -> None:
        self._prefill_scheduler.add_request(request)
        self._decode_scheduler.add_request(request)

    def is_empty(self) -> bool:
        return self._prefill_scheduler.is_empty() and self._decode_scheduler.is_empty()

    def schedule(self) -> List[Tuple[int, Request]]:
        prefill_requests = self._prefill_scheduler.schedule()
        decode_requests = self._decode_scheduler.schedule()

        return prefill_requests + decode_requests
