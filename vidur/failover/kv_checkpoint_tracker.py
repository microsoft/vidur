from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Tuple


class RequestCheckpoint:
    """KV checkpoint of one request, streamed from its replica to a holder's host memory."""

    def __init__(self, source_replica_id: int, holder_replica_id: int) -> None:
        self.source_replica_id = source_replica_id
        self.holder_replica_id = holder_replica_id
        # tokens handed to the network so far
        self.num_streamed_tokens = 0
        # (time the tokens become durable at the holder, checkpointed prefix length);
        # the head is always the longest prefix known to be durable
        self._durable: Deque[Tuple[float, int]] = deque()

    def add_stream(self, durable_at: float, num_tokens: int) -> None:
        self._durable.append((durable_at, num_tokens))
        self.num_streamed_tokens = num_tokens

    def get_num_durable_tokens(self, time: float) -> int:
        # simulation time never goes backwards, so older entries can be dropped
        while len(self._durable) > 1 and self._durable[1][0] <= time:
            self._durable.popleft()

        if self._durable and self._durable[0][0] <= time:
            return self._durable[0][1]

        return 0

    def reset(self) -> None:
        self.num_streamed_tokens = 0
        self._durable.clear()


class KVCheckpointTracker:
    """Models KV checkpoint streaming (DéjàVu-style) at token granularity.

    Each source replica streams new KV tokens over a single link of fixed
    bandwidth, so checkpoints lag behind the GPU-resident KV cache when the
    link is busy. A failure only preserves what was durable at the holder.
    """

    def __init__(
        self,
        num_replicas: int,
        kv_bytes_per_token: int,
        bandwidth_bytes_per_s: float,
        granularity_tokens: int,
    ) -> None:
        self._seconds_per_token = kv_bytes_per_token / bandwidth_bytes_per_s
        self._granularity_tokens = granularity_tokens

        self._checkpoints: Dict[int, RequestCheckpoint] = {}
        self._link_free_at = [0.0] * num_replicas
        # tokens of source replica requests checkpointed at each holder, i.e. the
        # KV a holder takes over if that source fails
        self._num_held_tokens: Dict[Tuple[int, int], int] = defaultdict(int)
        self._num_streamed_tokens = 0

    @property
    def num_streamed_tokens(self) -> int:
        return self._num_streamed_tokens

    def get(self, request_id: int) -> Optional[RequestCheckpoint]:
        return self._checkpoints.get(request_id)

    def get_num_held_tokens(
        self, source_replica_id: int, holder_replica_id: int
    ) -> int:
        return self._num_held_tokens[(source_replica_id, holder_replica_id)]

    def assign(
        self, request_id: int, source_replica_id: int, holder_replica_id: int
    ) -> RequestCheckpoint:
        self.drop(request_id)
        checkpoint = RequestCheckpoint(source_replica_id, holder_replica_id)
        self._checkpoints[request_id] = checkpoint
        return checkpoint

    def drop(self, request_id: int) -> None:
        checkpoint = self._checkpoints.pop(request_id, None)
        if checkpoint:
            self._release(checkpoint)

    def stream(
        self, time: float, checkpoint: RequestCheckpoint, num_processed_tokens: int
    ) -> None:
        num_tokens = (
            num_processed_tokens // self._granularity_tokens
        ) * self._granularity_tokens
        num_new_tokens = num_tokens - checkpoint.num_streamed_tokens
        if num_new_tokens <= 0:
            return

        source = checkpoint.source_replica_id
        start = max(time, self._link_free_at[source])
        self._link_free_at[source] = start + num_new_tokens * self._seconds_per_token

        checkpoint.add_stream(self._link_free_at[source], num_tokens)
        self._num_held_tokens[(source, checkpoint.holder_replica_id)] += num_new_tokens
        self._num_streamed_tokens += num_new_tokens

    def on_source_failed(self, replica_id: int, time: float) -> None:
        # transfers that had not completed are lost with the source
        self._link_free_at[replica_id] = time

    def on_holder_failed(self, replica_id: int) -> List[int]:
        request_ids = [
            request_id
            for request_id, checkpoint in self._checkpoints.items()
            if checkpoint.holder_replica_id == replica_id
        ]
        for request_id in request_ids:
            checkpoint = self._checkpoints[request_id]
            self._release(checkpoint)
            checkpoint.reset()
        return request_ids

    def _release(self, checkpoint: RequestCheckpoint) -> None:
        key = (checkpoint.source_replica_id, checkpoint.holder_replica_id)
        self._num_held_tokens[key] -= checkpoint.num_streamed_tokens
