"""Disaggregated prefill/decode (P/D) serving simulator.

Models a deployment with separate prefill replicas and decode replicas
(Splitwise / DistServe / Dynamo-style disaggregation).  A request is prefilled
on a prefill replica, its KV cache is shipped to a decode replica, and the
decode replica generates the remaining output tokens with continuous batching.

The goal is to answer questions such as "what prefill:decode replica split
minimises TTFT/TPOT for model M at load λ?" without running a full cluster.
Prefill and decode step times come from Vidur's trained execution time
predictor (see ``build_time_fns_from_predictor``), so the simulator reuses the
same per-operator profiling data as the rest of Vidur.

Modelling assumptions (kept deliberately simple, documented in
``docs/disaggregated_scheduling.md``):

* Prefill replicas process one request at a time, FIFO.  Prefill is compute
  bound, so batching prefills mostly trades latency for little throughput.
* Prefill completion produces the first output token (same convention as
  ``Request.on_batch_end``); TTFT is measured at prefill completion.
* Each prefill replica has one outbound link of ``kv_transfer_bandwidth_gb_per_s``
  (GB/s, aggregate over its tensor-parallel workers).  Transfers leaving the
  same prefill replica are serialised on that link.
* Decode replicas run continuous batching: every iteration produces one token
  for every running request, new requests join at iteration boundaries, and the
  running batch never exceeds ``max_decode_batch_size``.
* Requests wait in a single FIFO queue for any free decode slot.
"""

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np

from vidur.config import BaseExecutionTimePredictorConfig
from vidur.config.model_config import BaseModelConfig, BaseMoEModelConfig
from vidur.entities import Batch, Request
from vidur.execution_time_predictor import BaseExecutionTimePredictor

# (arrived_at, num_prefill_tokens, num_decode_tokens)
RequestSpec = Tuple[float, int, int]


def kv_cache_bytes_per_token(
    model_config: BaseModelConfig, kv_dtype_bytes: int = 2
) -> int:
    """KV cache bytes stored per token, summed over all layers.

    Models with Multi-head Latent Attention (``kv_lora_rank > 0``, e.g.
    DeepSeek-V3) cache one compressed latent vector plus the decoupled RoPE key
    per token instead of full per-head K and V.
    """
    if isinstance(model_config, BaseMoEModelConfig) and model_config.kv_lora_rank > 0:
        latent_dim = model_config.kv_lora_rank + model_config.qk_rope_head_dim
        return latent_dim * model_config.num_layers * kv_dtype_bytes

    head_dim = model_config.embedding_dim // model_config.num_q_heads
    return (
        2  # K and V
        * model_config.num_kv_heads
        * head_dim
        * model_config.num_layers
        * kv_dtype_bytes
    )


@dataclass
class DisaggregatedSimulationResult:
    num_requests: int
    makespan_s: float
    throughput_rps: float
    ttft_s: Dict[str, float]
    tpot_s: Dict[str, float]
    e2e_s: Dict[str, float]
    kv_transfer_s: Dict[str, float]
    prefill_utilization: float
    decode_utilization: float
    mean_decode_batch_size: float
    max_decode_batch_size_observed: int

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _percentiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0}
    arr = np.asarray(values)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
    }


@dataclass
class _DecodeReplica:
    # request id -> [kv tokens held, decode iterations still to run]
    running: Dict[int, List[int]] = field(default_factory=dict)
    busy: bool = False
    busy_time_s: float = 0.0


class DisaggregatedScheduler:
    """Discrete-event simulator for disaggregated prefill/decode serving.

    Parameters
    ----------
    num_prefill_replicas, num_decode_replicas:
        Size of each fleet.
    prefill_time_fn:
        ``(num_prefill_tokens) -> seconds`` for prefilling one request.
    decode_step_time_fn:
        ``(batch_size, avg_kv_tokens) -> seconds`` for one decode iteration.
    kv_bytes_per_token:
        KV cache bytes per token (see ``kv_cache_bytes_per_token``).
    kv_transfer_bandwidth_gb_per_s:
        Outbound KV transfer bandwidth of each prefill replica, in GB/s.
    max_decode_batch_size:
        Maximum number of concurrently running requests per decode replica.
    """

    def __init__(
        self,
        num_prefill_replicas: int,
        num_decode_replicas: int,
        prefill_time_fn: Callable[[int], float],
        decode_step_time_fn: Callable[[int, int], float],
        kv_bytes_per_token: int,
        kv_transfer_bandwidth_gb_per_s: float,
        max_decode_batch_size: int,
    ) -> None:
        assert num_prefill_replicas > 0 and num_decode_replicas > 0
        assert kv_transfer_bandwidth_gb_per_s > 0
        assert max_decode_batch_size > 0

        self._num_prefill_replicas = num_prefill_replicas
        self._num_decode_replicas = num_decode_replicas
        self._prefill_time_fn = prefill_time_fn
        self._decode_step_time_fn = decode_step_time_fn
        self._kv_bytes_per_token = kv_bytes_per_token
        self._bandwidth_bytes_per_s = kv_transfer_bandwidth_gb_per_s * 1e9
        self._max_decode_batch_size = max_decode_batch_size

    def kv_transfer_time_s(self, num_prefill_tokens: int) -> float:
        return (
            num_prefill_tokens * self._kv_bytes_per_token / self._bandwidth_bytes_per_s
        )

    def simulate(self, requests: List[RequestSpec]) -> DisaggregatedSimulationResult:
        requests = sorted(requests, key=lambda r: r[0])
        num_requests = len(requests)
        assert num_requests > 0

        events = []
        seq = itertools.count()

        def push(time: float, kind: str, payload) -> None:
            heapq.heappush(events, (time, next(seq), kind, payload))

        for request_id, (arrived_at, _, _) in enumerate(requests):
            push(arrived_at, "arrival", request_id)

        prefill_queue: List[int] = []
        prefill_free = list(range(self._num_prefill_replicas))
        prefill_busy_time_s = 0.0
        link_free_at = [0.0] * self._num_prefill_replicas

        decode_queue: List[int] = []
        decode_replicas = [_DecodeReplica() for _ in range(self._num_decode_replicas)]

        prefill_done_at = [0.0] * num_requests
        kv_transfer_s = [0.0] * num_requests
        completed_at = [0.0] * num_requests
        batch_size_samples: List[int] = []

        def dispatch_prefill(now: float) -> None:
            nonlocal prefill_busy_time_s
            while prefill_free and prefill_queue:
                replica_id = prefill_free.pop(0)
                request_id = prefill_queue.pop(0)
                duration = self._prefill_time_fn(requests[request_id][1])
                prefill_busy_time_s += duration
                push(now + duration, "prefill_done", (request_id, replica_id))

        def run_decode_iteration(now: float, replica_id: int) -> None:
            replica = decode_replicas[replica_id]
            while decode_queue and len(replica.running) < self._max_decode_batch_size:
                request_id = decode_queue.pop(0)
                _, num_prefill_tokens, num_decode_tokens = requests[request_id]
                # the first output token was produced by the prefill replica
                replica.running[request_id] = [
                    num_prefill_tokens + 1,
                    num_decode_tokens - 1,
                ]
            if not replica.running:
                replica.busy = False
                return
            batch_size = len(replica.running)
            avg_kv_tokens = int(
                sum(state[0] for state in replica.running.values()) / batch_size
            )
            duration = self._decode_step_time_fn(batch_size, avg_kv_tokens)
            replica.busy = True
            replica.busy_time_s += duration
            batch_size_samples.append(batch_size)
            push(now + duration, "decode_step_done", replica_id)

        def wake_idle_decode_replicas(now: float) -> None:
            for replica_id, replica in enumerate(decode_replicas):
                if not decode_queue:
                    return
                if not replica.busy:
                    run_decode_iteration(now, replica_id)

        now = 0.0
        while events:
            now, _, kind, payload = heapq.heappop(events)

            if kind == "arrival":
                prefill_queue.append(payload)
                dispatch_prefill(now)

            elif kind == "prefill_done":
                request_id, replica_id = payload
                prefill_done_at[request_id] = now
                prefill_free.append(replica_id)
                dispatch_prefill(now)

                if requests[request_id][2] <= 1:
                    completed_at[request_id] = now
                    continue
                start = max(now, link_free_at[replica_id])
                end = start + self.kv_transfer_time_s(requests[request_id][1])
                link_free_at[replica_id] = end
                kv_transfer_s[request_id] = end - now
                push(end, "kv_transfer_done", request_id)

            elif kind == "kv_transfer_done":
                decode_queue.append(payload)
                wake_idle_decode_replicas(now)

            elif kind == "decode_step_done":
                replica = decode_replicas[payload]
                for request_id in list(replica.running):
                    state = replica.running[request_id]
                    state[0] += 1
                    state[1] -= 1
                    if state[1] == 0:
                        completed_at[request_id] = now
                        del replica.running[request_id]
                run_decode_iteration(now, payload)

        makespan_s = max(completed_at) - requests[0][0]

        ttft = [prefill_done_at[i] - requests[i][0] for i in range(num_requests)]
        e2e = [completed_at[i] - requests[i][0] for i in range(num_requests)]
        tpot = [
            (completed_at[i] - prefill_done_at[i]) / (requests[i][2] - 1)
            for i in range(num_requests)
            if requests[i][2] > 1
        ]
        transferred = [
            kv_transfer_s[i] for i in range(num_requests) if requests[i][2] > 1
        ]

        return DisaggregatedSimulationResult(
            num_requests=num_requests,
            makespan_s=makespan_s,
            throughput_rps=num_requests / makespan_s if makespan_s > 0 else 0.0,
            ttft_s=_percentiles(ttft),
            tpot_s=_percentiles(tpot),
            e2e_s=_percentiles(e2e),
            kv_transfer_s=_percentiles(transferred),
            prefill_utilization=prefill_busy_time_s
            / (makespan_s * self._num_prefill_replicas),
            decode_utilization=sum(r.busy_time_s for r in decode_replicas)
            / (makespan_s * self._num_decode_replicas),
            mean_decode_batch_size=(
                float(np.mean(batch_size_samples)) if batch_size_samples else 0.0
            ),
            max_decode_batch_size_observed=max(batch_size_samples, default=0),
        )


def build_time_fns_from_predictor(
    execution_time_predictor: BaseExecutionTimePredictor,
    predictor_config: BaseExecutionTimePredictorConfig,
) -> Tuple[Callable[[int], float], Callable[[int, int], float]]:
    """Wrap Vidur's execution time predictor as P/D step-time functions.

    Builds single-stage batches with the same request state transitions the
    replica schedulers use (``Request.on_batch_end``).  Decode batches are
    synthesised with every request at the batch-average KV length, which is
    exactly the quantity the sklearn predictors key their attention model on.
    """
    granularity = predictor_config.kv_cache_prediction_granularity
    prefill_cache: Dict[int, float] = {}
    decode_cache: Dict[Tuple[int, int], float] = {}

    def prefill_time_fn(num_prefill_tokens: int) -> float:
        if num_prefill_tokens not in prefill_cache:
            request = Request(0, num_prefill_tokens, 1)
            batch = Batch(0, [request], [num_prefill_tokens])
            prefill_cache[num_prefill_tokens] = (
                execution_time_predictor.get_execution_time(batch, 0).total_time
            )
        return prefill_cache[num_prefill_tokens]

    def decode_step_time_fn(batch_size: int, avg_kv_tokens: int) -> float:
        kv_tokens = max(granularity, -(-avg_kv_tokens // granularity) * granularity)
        key = (batch_size, kv_tokens)
        if key not in decode_cache:
            decode_requests = []
            for _ in range(batch_size):
                # prompt of kv_tokens - 1; prefill completion adds the first token
                request = Request(0, kv_tokens - 1, 2)
                request.on_batch_end(0, kv_tokens - 1)
                decode_requests.append(request)
            batch = Batch(0, decode_requests, [1] * batch_size)
            decode_cache[key] = execution_time_predictor.get_execution_time(
                batch, 0
            ).total_time
        return decode_cache[key]

    return prefill_time_fn, decode_step_time_fn
