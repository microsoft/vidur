from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from vidur.config import InfaasGlobalSchedulerConfig, SimulationConfig
from vidur.entities import Batch, Replica, Request
from vidur.execution_time_predictor import ExecutionTimePredictorRegistry
from vidur.logger import init_logger
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler

logger = init_logger(__name__)


class ReplicaHealth(Enum):
    ACTIVE = "active"
    OVERLOADED = "overloaded"
    INTERFERED = "interfered"
    INACTIVE = "inactive"


@dataclass
class ReplicaStatus:
    ewma_latency_ms: Optional[float] = None
    num_recent_violations: int = 0
    overload_cooldown_remaining: int = 0
    interference_cooldown_remaining: int = 0
    state: ReplicaHealth = ReplicaHealth.ACTIVE


class InfaasGlobalScheduler(BaseGlobalScheduler):
    """
    INFaaS-style global scheduler that tracks per-replica latency/queue state and
    routes requests based on a weighted cost model.
    """

    def __init__(self, config: SimulationConfig, replicas: Dict[int, Replica]) -> None:
        super().__init__(config, replicas)

        gcfg: InfaasGlobalSchedulerConfig = (
            config.cluster_config.global_scheduler_config
        )

        self._alpha = gcfg.alpha
        self._beta = gcfg.beta
        self._gamma = gcfg.gamma
        self._target_latency_ms = gcfg.target_latency_ms
        self._ewma_alpha = gcfg.ewma_alpha
        self._overload_latency_factor = gcfg.overload_latency_factor
        self._interference_latency_factor = gcfg.interference_latency_factor
        self._overload_cooldown = gcfg.overload_cooldown
        self._interference_cooldown = gcfg.interference_cooldown

        self._overload_threshold_ms = (
            self._target_latency_ms * self._overload_latency_factor
        )
        self._interference_threshold_ms = (
            self._target_latency_ms * self._interference_latency_factor
        )

        # Queue depth thresholds can be overridden if present on the config
        self._queue_depth_threshold = gcfg.queue_depth_threshold
        self._interference_queue_threshold = gcfg.interference_queue_threshold

        self._num_priority_levels = getattr(
            config.request_generator_config, "num_priority_levels", 1
        )

        # Predict service time for the cost model
        self._execution_time_predictor = ExecutionTimePredictorRegistry.get(
            config.execution_time_predictor_config.get_type(),
            predictor_config=config.execution_time_predictor_config,
            replica_config=config.cluster_config.replica_config,
            replica_scheduler_config=config.cluster_config.replica_scheduler_config,
            metrics_config=config.metrics_config,
        )

        self._replica_stats: Dict[int, ReplicaStatus] = {
            rid: ReplicaStatus() for rid in replicas
        }
        self._inflight_requests: Dict[int, Tuple[int, Request]] = {}

    def schedule(self) -> List[Tuple[int, Request]]:
        if not self._request_queue:
            self._tick_cooldowns()
            self._update_replica_states_from_completions()
            return []

        self.sort_requests()
        self._tick_cooldowns()
        self._update_replica_states_from_completions()

        assignments: List[Tuple[int, Request]] = []

        while self._request_queue:
            request = self._request_queue.pop(0)
            replica_id = self._choose_replica(request)
            if replica_id is None:
                # Fallback to the first replica if no candidates found
                replica_id = next(iter(self._replica_schedulers.keys()))

            assignments.append((replica_id, request))
            self._inflight_requests[request.id] = (replica_id, request)

        return assignments

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------
    def _choose_replica(self, request: Request) -> Optional[int]:
        candidates = self._eligible_replicas()
        if not candidates:
            return None

        best_cost = float("inf")
        best_rid: Optional[int] = None
        priority_scale = self._priority_scale(request)

        for rid in candidates:
            stats = self._replica_stats[rid]
            queue_component = self._estimate_queue_depth(rid) * priority_scale
            service_component = (
                self._predict_service_time_ms(request, rid) * priority_scale
            )

            penalty = 0.0
            if stats.state == ReplicaHealth.OVERLOADED:
                penalty = 1.0
            elif stats.state == ReplicaHealth.INTERFERED:
                penalty = 0.5
            elif stats.state == ReplicaHealth.INACTIVE:
                penalty = 2.0

            cost = (
                self._alpha * queue_component
                + self._beta * service_component
                + self._gamma * penalty
            )

            if cost < best_cost or (cost == best_cost and best_rid is not None and rid < best_rid):
                best_cost = cost
                best_rid = rid

        return best_rid

    def _eligible_replicas(self) -> List[int]:
        active = [
            rid for rid, stats in self._replica_stats.items() if stats.state == ReplicaHealth.ACTIVE
        ]
        if active:
            return active

        fallback = [
            rid
            for rid, stats in self._replica_stats.items()
            if stats.state != ReplicaHealth.INACTIVE
        ]
        return fallback

    def _estimate_queue_depth(self, replica_id: int) -> int:
        scheduler = self._replica_schedulers.get(replica_id)
        if not scheduler:
            return 0

        try:
            pending = scheduler.num_pending_requests
        except Exception:
            pending = len(getattr(scheduler, "_request_queue", []))

        running = len(getattr(scheduler, "_allocation_map", {}))
        preempted_requests = len(getattr(scheduler, "_preempted_requests", []))
        preempted_batches = len(getattr(scheduler, "_preempted_batches", []))

        return pending + running + preempted_requests + preempted_batches

    def _predict_service_time_ms(self, request: Request, replica_id: int) -> float:
        next_tokens = (
            request.num_prefill_tokens if not request.is_prefill_complete else 1
        )
        batch = Batch(replica_id, [request], [next_tokens])
        exec_time = self._execution_time_predictor.get_execution_time(batch, 0)
        return exec_time.total_time * 1e3

    def _priority_scale(self, request: Request) -> float:
        priority = max(0, int(getattr(request, "priority", 0)))
        if self._num_priority_levels <= 1:
            return 1.0

        # Higher priority (smaller number) → smaller multiplier
        distance_from_top = max(0, self._num_priority_levels - 1 - priority)
        scale = 1.0 - 0.1 * distance_from_top
        return max(0.5, scale)

    def _update_replica_states_from_completions(self) -> None:
        completed_request_ids = []

        for req_id, (rid, request) in list(self._inflight_requests.items()):
            if not request.completed:
                continue

            completed_request_ids.append(req_id)
            latency_ms = self._request_latency_ms(request)
            stats = self._replica_stats[rid]

            if latency_ms > self._target_latency_ms:
                stats.num_recent_violations += 1
            elif stats.num_recent_violations > 0:
                stats.num_recent_violations -= 1

            if stats.ewma_latency_ms is None:
                stats.ewma_latency_ms = latency_ms
            else:
                stats.ewma_latency_ms = (
                    self._ewma_alpha * latency_ms
                    + (1 - self._ewma_alpha) * stats.ewma_latency_ms
                )

            queue_depth = self._estimate_queue_depth(rid)
            is_overloaded = (
                stats.ewma_latency_ms > self._overload_threshold_ms
                and queue_depth > self._queue_depth_threshold
            )
            is_interfered = (
                stats.ewma_latency_ms > self._interference_threshold_ms
                and queue_depth <= self._interference_queue_threshold
            )

            if is_overloaded:
                stats.state = ReplicaHealth.OVERLOADED
                stats.overload_cooldown_remaining = self._overload_cooldown
            elif is_interfered:
                stats.state = ReplicaHealth.INTERFERED
                stats.interference_cooldown_remaining = self._interference_cooldown
            else:
                stats.state = ReplicaHealth.ACTIVE

        for req_id in completed_request_ids:
            self._inflight_requests.pop(req_id, None)

    def _request_latency_ms(self, request: Request) -> float:
        try:
            latency = request.completed_at - request.scheduled_at
        except Exception:
            latency = request.completed_at - request.arrived_at
        return max(latency * 1e3, 0.0)

    def _tick_cooldowns(self) -> None:
        for rid, stats in self._replica_stats.items():
            if stats.overload_cooldown_remaining > 0:
                stats.overload_cooldown_remaining -= 1
            if stats.interference_cooldown_remaining > 0:
                stats.interference_cooldown_remaining -= 1

            if (
                stats.state == ReplicaHealth.OVERLOADED
                and stats.overload_cooldown_remaining == 0
            ):
                if self._can_recover(rid, stats, self._overload_threshold_ms, self._queue_depth_threshold):
                    stats.state = ReplicaHealth.ACTIVE

            if (
                stats.state == ReplicaHealth.INTERFERED
                and stats.interference_cooldown_remaining == 0
            ):
                if self._can_recover(
                    rid, stats, self._interference_threshold_ms, self._interference_queue_threshold
                ):
                    stats.state = ReplicaHealth.ACTIVE

    def _can_recover(
        self,
        replica_id: int,
        stats: ReplicaStatus,
        latency_threshold: float,
        queue_threshold: int,
    ) -> bool:
        if stats.ewma_latency_ms is None:
            return True

        queue_depth = self._estimate_queue_depth(replica_id)
        if stats.ewma_latency_ms <= latency_threshold:
            return True

        return queue_depth <= queue_threshold
