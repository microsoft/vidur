"""Disaggregated prefill/decode scheduler for VIDUR.
# ruff: noqa: E402
from __future__ import annotations  # defer annotation evaluation for dataclasses


Models separate prefill-worker and decode-worker fleets (Dynamo-style
disaggregated serving).  The key addition over the existing collocated
scheduler is the **KV transfer latency**: after a prefill worker completes a
request, the generated KV cache must be transferred to a decode worker before
decoding can begin.

This scheduler extends VIDUR's simulation to answer the question:
    "What is the optimal prefill:decode worker ratio for model M at traffic λ?"

Usage
-----
Add ``--scheduler_type disaggregated`` to the VIDUR launch command.
``DisaggregatedReplicaSchedulerConfig`` exposes:
- ``prefill_fleet_size``: number of prefill replicas
- ``decode_fleet_size``: number of decode replicas
- ``interconnect_bandwidth_gbps``: NVLink (600), InfiniBand (400), PCIe (64)

The P/D ratio ``prefill_fleet_size / (prefill_fleet_size + decode_fleet_size)``
is what Vidur-Search sweeps to find the optimum.

Architecture notes
------------------
The scheduler runs as a discrete-event simulation.  Each simulation tick
advances the clock to the next event (request arrival, prefill completion, KV
transfer completion, or decode completion).  The queues are:

    arrival_queue → prefill_queue → kv_transfer_queue → decode_queue → done

KV transfer latency is modelled as:

    t_transfer = kv_bytes / bandwidth
    kv_bytes   = 2 * num_layers * num_kv_heads * kv_head_dim * seq_len * 2  (fp16)
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from vidur.config import ReplicaConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KV transfer helper
# ---------------------------------------------------------------------------


def kv_cache_bytes_per_token(replica_config: ReplicaConfig) -> float:
    """Return the KV cache size in bytes for a single token."""
    model_cfg = replica_config.model_config
    # KV cache: 2 tensors (K + V), num_kv_heads heads, head_dim features, fp16
    head_dim = model_cfg.embedding_dim // model_cfg.num_q_heads
    return (
        2  # K + V
        * model_cfg.num_kv_heads
        * head_dim
        * model_cfg.num_layers
        * 2  # bytes per fp16
    )


def kv_transfer_latency_s(
    seq_len: int,
    replica_config: ReplicaConfig,
    bandwidth_gbps: float,
) -> float:
    """Return the KV transfer wall-clock time in seconds."""
    bytes_per_token = kv_cache_bytes_per_token(replica_config)
    total_bytes = bytes_per_token * seq_len
    return total_bytes / (bandwidth_gbps * 1e9)


# ---------------------------------------------------------------------------
# Simulator events
# ---------------------------------------------------------------------------


@dataclass(order=True)
class _Event:
    time: float
    kind: str = field(compare=False)
    payload: object = field(compare=False, default=None)


# ---------------------------------------------------------------------------
# Main scheduler
# ---------------------------------------------------------------------------


@dataclass
class DisaggregatedSimulationResult:
    """Summary statistics from a disaggregated simulation run."""

    num_requests: int
    prefill_fleet_size: int
    decode_fleet_size: int
    interconnect_bandwidth_gbps: float

    p50_e2e_latency_s: float
    p90_e2e_latency_s: float
    p99_e2e_latency_s: float
    mean_e2e_latency_s: float

    p50_ttft_s: float  # time-to-first-token
    p90_ttft_s: float
    mean_ttft_s: float

    p50_kv_transfer_s: float
    mean_kv_transfer_s: float

    prefill_utilisation: float
    decode_utilisation: float
    throughput_rps: float


class DisaggregatedScheduler:
    """Discrete-event simulator for disaggregated P/D serving.

    Parameters
    ----------
    replica_config:
        VIDUR replica config (used for model dimensions to compute KV size).
    prefill_fleet_size:
        Number of prefill workers.
    decode_fleet_size:
        Number of decode workers.
    interconnect_bandwidth_gbps:
        Interconnect bandwidth for KV cache transfer.
        Typical values: NVLink 600, InfiniBand 400, PCIe 64.
    prefill_time_fn:
        Callable ``(seq_len: int) -> float`` returning prefill time in seconds
        (use VIDUR's existing execution-time predictor).
    decode_time_fn:
        Callable ``(seq_len: int, output_len: int) -> float`` returning total
        decode time in seconds.
    """

    def __init__(
        self,
        replica_config: ReplicaConfig,
        prefill_fleet_size: int,
        decode_fleet_size: int,
        interconnect_bandwidth_gbps: float,
        prefill_time_fn,
        decode_time_fn,
    ) -> None:
        self._replica_config = replica_config
        self._prefill_fleet_size = prefill_fleet_size
        self._decode_fleet_size = decode_fleet_size
        self._bandwidth_gbps = interconnect_bandwidth_gbps
        self._prefill_time_fn = prefill_time_fn
        self._decode_time_fn = decode_time_fn

    def simulate(
        self,
        requests: List[
            Tuple[float, int, int]
        ],  # (arrival_time, prefill_len, output_len)
    ) -> DisaggregatedSimulationResult:
        """Run the simulation and return aggregated statistics.

        Parameters
        ----------
        requests:
            List of ``(arrival_time_s, prompt_len, output_len)`` tuples.
        """
        import numpy as np

        n = len(requests)
        if n == 0:
            raise ValueError("No requests to simulate")

        # Event heap
        heap: List[_Event] = []

        prefill_workers_free = self._prefill_fleet_size
        decode_workers_free = self._decode_fleet_size

        prefill_queue: List[Tuple[float, int, int, int]] = (
            []
        )  # (arrival, prefill_len, output_len, req_id)
        kv_queue: List[Tuple[float, int, int, int]] = (
            []
        )  # (ready_time, prefill_len, output_len, req_id)
        decode_queue: List[Tuple[float, int, int, int]] = []  # same

        arrival_times = {}  # req_id -> arrival_time
        prefill_done = {}  # req_id -> prefill completion time
        kv_done = {}  # req_id -> kv transfer done time
        decode_done = {}  # req_id -> decode done time

        prefill_busy_time = 0.0
        decode_busy_time = 0.0

        # Schedule arrivals
        for req_id, (arr, plen, olen) in enumerate(requests):
            heapq.heappush(heap, _Event(arr, "arrival", (req_id, arr, plen, olen)))
            arrival_times[req_id] = arr

        def try_dispatch_prefill(now: float):
            nonlocal prefill_workers_free
            while prefill_workers_free > 0 and prefill_queue:
                _, plen, olen, req_id = prefill_queue.pop(0)
                prefill_workers_free -= 1
                t_prefill = self._prefill_time_fn(plen)
                t_done = now + t_prefill
                prefill_busy_time_container[0] += t_prefill
                heapq.heappush(
                    heap, _Event(t_done, "prefill_done", (req_id, plen, olen, t_done))
                )

        def try_dispatch_kv(now: float):
            while kv_queue:
                _, plen, olen, req_id = kv_queue.pop(0)
                t_kv = kv_transfer_latency_s(
                    plen, self._replica_config, self._bandwidth_gbps
                )
                t_done = now + t_kv
                heapq.heappush(
                    heap, _Event(t_done, "kv_done", (req_id, plen, olen, t_done))
                )

        def try_dispatch_decode(now: float):
            nonlocal decode_workers_free
            while decode_workers_free > 0 and decode_queue:
                _, plen, olen, req_id = decode_queue.pop(0)
                decode_workers_free -= 1
                t_decode = self._decode_time_fn(plen, olen)
                t_done = now + t_decode
                decode_busy_time_container[0] += t_decode
                heapq.heappush(heap, _Event(t_done, "decode_done", (req_id, t_done)))

        prefill_busy_time_container = [0.0]
        decode_busy_time_container = [0.0]

        completed = 0
        sim_end = 0.0

        while heap and completed < n:
            ev = heapq.heappop(heap)
            now = ev.time
            sim_end = max(sim_end, now)

            if ev.kind == "arrival":
                req_id, arr, plen, olen = ev.payload
                prefill_queue.append((arr, plen, olen, req_id))
                try_dispatch_prefill(now)

            elif ev.kind == "prefill_done":
                nonlocal_prefill_free = True
                req_id, plen, olen, t_done = ev.payload
                prefill_workers_free += 1
                prefill_done[req_id] = t_done
                kv_queue.append((t_done, plen, olen, req_id))
                try_dispatch_prefill(now)
                try_dispatch_kv(now)

            elif ev.kind == "kv_done":
                req_id, plen, olen, t_done = ev.payload
                kv_done[req_id] = t_done
                decode_queue.append((t_done, plen, olen, req_id))
                try_dispatch_decode(now)

            elif ev.kind == "decode_done":
                req_id, t_done = ev.payload
                decode_workers_free += 1
                decode_done[req_id] = t_done
                completed += 1
                try_dispatch_decode(now)

        # Compute statistics
        e2e_latencies = [
            decode_done[i] - arrival_times[i] for i in range(n) if i in decode_done
        ]
        ttfts = [kv_done[i] - arrival_times[i] for i in range(n) if i in kv_done]
        kv_latencies = [
            kv_done[i] - prefill_done[i]
            for i in range(n)
            if i in kv_done and i in prefill_done
        ]

        e2e = np.array(e2e_latencies)
        ttft = np.array(ttfts)
        kv_lat = np.array(kv_latencies) if kv_latencies else np.array([0.0])

        total_sim_time = sim_end - requests[0][0] if sim_end > requests[0][0] else 1.0
        prefill_util = prefill_busy_time_container[0] / (
            self._prefill_fleet_size * total_sim_time + 1e-9
        )
        decode_util = decode_busy_time_container[0] / (
            self._decode_fleet_size * total_sim_time + 1e-9
        )

        return DisaggregatedSimulationResult(
            num_requests=n,
            prefill_fleet_size=self._prefill_fleet_size,
            decode_fleet_size=self._decode_fleet_size,
            interconnect_bandwidth_gbps=self._bandwidth_gbps,
            p50_e2e_latency_s=float(np.percentile(e2e, 50)),
            p90_e2e_latency_s=float(np.percentile(e2e, 90)),
            p99_e2e_latency_s=float(np.percentile(e2e, 99)),
            mean_e2e_latency_s=float(e2e.mean()),
            p50_ttft_s=float(np.percentile(ttft, 50)),
            p90_ttft_s=float(np.percentile(ttft, 90)),
            mean_ttft_s=float(ttft.mean()),
            p50_kv_transfer_s=float(np.percentile(kv_lat, 50)),
            mean_kv_transfer_s=float(kv_lat.mean()),
            prefill_utilisation=float(prefill_util),
            decode_utilisation=float(decode_util),
            throughput_rps=float(len(e2e_latencies) / total_sim_time),
        )
