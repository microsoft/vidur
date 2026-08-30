from typing import Dict, List, Tuple, Optional
import math
from vidur.config import SimulationConfig
from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler
from vidur.scheduler.replica_scheduler.llumlet_replica_scheduler import LlumletLocalScheduler
from vidur.execution_time_predictor import ExecutionTimePredictorRegistry
from vidur.logger import init_logger

logger = init_logger(__name__)


class LlumnixGlobalScheduler(BaseGlobalScheduler):
    """
    Llumnix-style GLOBAL scheduler (faithful policy):
      - Does NOT inspect per-request internals beyond public llumlet API.
      - Uses llumlet-reported freeness F = (M - ΣV) / B; negative allowed.
      - Dispatches to the freest instance (no hard capacity gate).
      - Periodically pairs overloaded/underloaded instances; llumlets choose the request and run a live-migration handshake.
      - Exposes autoscale recommendations via average freeness bands.
    """

    def __init__(self, config: SimulationConfig, replicas) -> None:
        # Manually set up base fields
        super().__init__(config, replicas)
        self._config = config
        self._replicas = replicas
        self._num_replicas = len(replicas)
        self._request_queue: List[Request] = []

        # Predictors (as BaseGlobalScheduler would)
        execution_time_predictor = ExecutionTimePredictorRegistry.get(
            config.execution_time_predictor_config.get_type(),
            predictor_config=config.execution_time_predictor_config,
            replica_config=config.cluster_config.replica_config,
            replica_scheduler_config=config.cluster_config.replica_scheduler_config,
            metrics_config=config.metrics_config,
        )

        # Instantiate Llumlet per replica
        self._replica_schedulers: Dict[int, LlumletLocalScheduler] = {
            rid: LlumletLocalScheduler(
                config.cluster_config.replica_config,
                config.cluster_config.replica_scheduler_config,
                config.request_generator_config,
                replica,
                replica.num_pipeline_stages,
                execution_time_predictor,
            )
            for rid, replica in replicas.items()
        }

        # Llumnix-specific tuning
        gcfg = config.cluster_config.global_scheduler_config
        self._enable_migration = getattr(gcfg, "enable_migration", True)
        self._rebalance_interval = getattr(gcfg, "rebalance_interval", 0.5)
        self._last_rebalance_time = 0.0
        self._num_priority_levels = getattr(gcfg, "num_priority_levels", 3)
        # Use gap on freeness to trigger rebalancing
        self._load_imbalance_threshold = getattr(gcfg, "load_imbalance_threshold", 0.5)
        self._src_freeness_threshold = getattr(gcfg, "src_freeness_threshold", None)  # dynamic if None
        self._dst_freeness_threshold = getattr(gcfg, "dst_freeness_threshold", None)

        # Autoscale bands (avg F): scale_out if below low; scale_in if above high
        self._autoscale_low = getattr(gcfg, "autoscale_low", -0.5)
        self._autoscale_high = getattr(gcfg, "autoscale_high", 1.5)

        self._migration_count = 0

    # -------------------- Helpers (with fallback for non-llumlet schedulers) --------------------
    def _all_freeness(self) -> List[Tuple[int, float]]:
        """Get freeness from all replicas. Falls back to simple metric for non-llumlet schedulers."""
        result = []
        for rid, sch in self._replica_schedulers.items():
            if hasattr(sch, 'report_freeness'):
                result.append((rid, sch.report_freeness()))
            else:
                # Fallback: estimate freeness from free blocks / batch size
                free_blocks = getattr(sch, '_num_free_blocks', 0)
                batch_size = getattr(sch, '_batch_size', 1)
                result.append((rid, float(free_blocks) / max(1, batch_size)))
        return result

    def _all_normal_priority_freeness(self) -> List[Tuple[int, float]]:
        """
        Get freeness for normal-priority requests only (excludes high-priority headroom).
        Per paper Section 4.4.3, Algorithm 1 line 17: autoscaling uses normal-priority freeness.
        Falls back to regular freeness if not supported.
        """
        result = []
        for rid, sch in self._replica_schedulers.items():
            if hasattr(sch, 'report_normal_priority_freeness'):
                result.append((rid, sch.report_normal_priority_freeness()))
            elif hasattr(sch, 'report_freeness'):
                # Fallback: use regular freeness (no priority distinction)
                result.append((rid, sch.report_freeness()))
            else:
                # Fallback: estimate from free blocks
                free_blocks = getattr(sch, '_num_free_blocks', 0)
                batch_size = getattr(sch, '_batch_size', 1)
                result.append((rid, float(free_blocks) / max(1, batch_size)))
        return result

    def _all_running_request_counts(self) -> List[Tuple[int, int]]:
        """
        Get running request counts for each replica.
        Per paper Section 4.4.3: scale-in selects "instance with fewest running requests".
        """
        result = []
        for rid, sch in self._replica_schedulers.items():
            if hasattr(sch, '_allocation_map'):
                result.append((rid, len(sch._allocation_map)))
            elif hasattr(sch, '_running_requests'):
                result.append((rid, len(sch._running_requests)))
            else:
                # Fallback: assume 0 running requests
                result.append((rid, 0))
        return result

    def _freest_rid(self) -> Optional[int]:
        best = None
        best_F = -float("inf")
        for rid, sch in self._replica_schedulers.items():
            if hasattr(sch, 'report_freeness'):
                F = sch.report_freeness()
            else:
                # Fallback: estimate freeness
                free_blocks = getattr(sch, '_num_free_blocks', 0)
                batch_size = getattr(sch, '_batch_size', 1)
                F = float(free_blocks) / max(1, batch_size)
            
            if F > best_F:
                best_F, best = F, rid
        return best

    def _imbalance_gap(self) -> float:
        Fs = [F for _, F in self._all_freeness()]
        if not Fs:
            return 0.0
        return (max(Fs) - min(Fs))

    # -------------------- New Request Placement (priority-aware) --------------------
    def schedule(self) -> List[Tuple[int, Request]]:
        """
        Llumnix-compliant request placement:
        • never place new requests on draining replicas
        • choose among non-draining replicas with highest freeness
        • if all replicas are draining, place on the least-bad (highest freeness)
        • preserve priority ordering
        """
        if not self._request_queue:
            return []

        # --- Group by priority (0 = highest) ---
        by_pr: Dict[int, List[Request]] = {}
        for req in self._request_queue:
            pr = getattr(req, "priority", 0)
            by_pr.setdefault(pr, []).append(req)
        self._request_queue.clear()

        assignments: List[Tuple[int, Request]] = []

        # Sort priority buckets: low number = high priority
        for pr in sorted(by_pr.keys()):
            for req in by_pr[pr]:

                # 1. Select target replica among *non-draining* ones
                candidates = []
                for rid, sch in self._replica_schedulers.items():
                    is_draining = getattr(sch, '_is_draining', False)
                    if not is_draining:
                        if hasattr(sch, 'report_freeness'):
                            F = sch.report_freeness()
                        else:
                            # Fallback: estimate freeness
                            free_blocks = getattr(sch, '_num_free_blocks', 0)
                            batch_size = getattr(sch, '_batch_size', 1)
                            F = float(free_blocks) / max(1, batch_size)
                        candidates.append((rid, F))

                if not candidates:
                    # Fallback: all replicas are draining → place on best available
                    for rid, sch in self._replica_schedulers.items():
                        if hasattr(sch, 'report_freeness'):
                            F = sch.report_freeness()
                        else:
                            free_blocks = getattr(sch, '_num_free_blocks', 0)
                            batch_size = getattr(sch, '_batch_size', 1)
                            F = float(free_blocks) / max(1, batch_size)
                        candidates.append((rid, F))

                # Pick replica with max F
                rid = max(candidates, key=lambda x: x[1])[0]

                # Add to assignments (GlobalScheduleEvent will call add_request)
                assignments.append((rid, req))

        return assignments


    # -------------------- Migration Triggering --------------------
    def should_rebalance(self, now: float) -> bool:
        if not self._enable_migration or self._num_replicas < 2:
            return False
        if (now - self._last_rebalance_time) < self._rebalance_interval:
            return False
        return self._imbalance_gap() >= self._load_imbalance_threshold


    def rebalance(self, now: float) -> List[Tuple[int, int, int]]:
        """
        Returns list of (req_id, src_rid, dst_rid) migrations.
        """
        self._last_rebalance_time = now
        migrations = []

        freeness = sorted(self._all_freeness(), key=lambda x: x[1])
        if len(freeness) < 2:
            return migrations

        minF = freeness[0][1]
        maxF = freeness[-1][1]
        if (maxF - minF) < self._load_imbalance_threshold:
            return migrations

        # dynamic thresholds if user doesn't specify
        src_thresh = self._src_freeness_threshold or (minF + 0.1)
        dst_thresh = self._dst_freeness_threshold or (maxF - 0.1)

        # -------------------------------
        # Sources: overloaded OR draining
        # -------------------------------
        sources = []
        for rid, F in freeness:
            sch = self._replica_schedulers[rid]

            if sch._is_draining:
                # draining replica evacuates only if it has any work
                if sch._priority_queue or sch._allocation_map:
                    sources.append((rid, F))

            elif F <= src_thresh:
                # overloaded replica
                sources.append((rid, F))

        # -------------------------------
        # Destinations: underloaded, not draining
        # -------------------------------
        dests = [
            (rid, F)
            for rid, F in reversed(freeness)
            if (F >= dst_thresh) and not self._replica_schedulers[rid]._is_draining
        ]

        # -------------------------------
        # Pair sources → dests
        # -------------------------------
        for (src_rid, _), (dst_rid, _) in zip(sources, dests):

            if src_rid == dst_rid:
                continue

            src = self._replica_schedulers[src_rid]
            dst = self._replica_schedulers[dst_rid]

            # redundant safety check
            if getattr(dst, '_is_draining', False):
                continue

            # Check if replica schedulers support migration
            if not hasattr(src, 'begin_migration_to') or not hasattr(dst, 'begin_migration_to'):
                # Migration not supported by this replica scheduler type
                continue

            # CRITICAL: When source is DRAINING, migrate ALL requests, not just one
            # Keep calling begin_migration_to() until source is empty or destination full
            is_draining = getattr(src, '_is_draining', False)
            mig_count = 0
            while True:
                mig = src.begin_migration_to(dst)
                if mig:
                    migrations.append(mig)
                    self._migration_count += 1
                    mig_count += 1
                else:
                    # No more migrations available
                    break
                
                # If not draining, only do one migration per rebalance
                if not is_draining:
                    break
            
            # Log drain migrations (they happen in batches)
            if is_draining and mig_count > 0:
                logger.info(
                    f"[Llumnix Drain] Replica {src_rid} → {dst_rid}: "
                    f"migrated {mig_count} requests (draining)"
                )

        return migrations



    # -------------------- Autoscaling signal --------------------
    def autoscale_recommendation(self) -> Optional[str]:
        """
        Paper-compliant autoscaling: uses normal-priority freeness only.
        Per Section 4.4.3, Algorithm 1 line 17: "average freeness for the normal priority".
        """
        Fs = [F for _, F in self._all_normal_priority_freeness()]
        if not Fs:
            return None
        avgF = sum(Fs) / len(Fs)
        if avgF < self._autoscale_low:
            return "scale_out"
        if avgF > self._autoscale_high:
            return "scale_in"
        return None


    def set_draining(self, replica_ids: List[int], draining: bool = True) -> None:
        """Mark replicas as draining (or not). Llumnix-compliant scale-in."""
        for rid in replica_ids:
            if rid in self._replica_schedulers:
                sch = self._replica_schedulers[rid]
                if hasattr(sch, '_is_draining'):
                    old_val = sch._is_draining
                    sch._is_draining = draining
                    logger.info(
                        f"[GlobalScheduler] Replica {rid} draining flag changed: {old_val} → {draining}"
                    )

    def add_replica(self, replica) -> int:
        """
        Dynamically add a new replica to the global scheduler for scale-out.
        
        Args:
            replica: The Replica instance to add
            
        Returns:
            The replica ID of the newly added replica
        """
        rid = replica.id
        
        # Add to replicas dict
        self._replicas[rid] = replica
        self._num_replicas = len(self._replicas)
        
        # Create execution time predictor for this replica
        execution_time_predictor = ExecutionTimePredictorRegistry.get(
            self._config.execution_time_predictor_config.get_type(),
            predictor_config=self._config.execution_time_predictor_config,
            replica_config=self._config.cluster_config.replica_config,
            replica_scheduler_config=self._config.cluster_config.replica_scheduler_config,
            metrics_config=self._config.metrics_config,
        )
        
        # Create replica scheduler for the new replica
        self._replica_schedulers[rid] = LlumletLocalScheduler(
            self._config.cluster_config.replica_config,
            self._config.cluster_config.replica_scheduler_config,
            self._config.request_generator_config,
            replica,
            replica.num_pipeline_stages,
            execution_time_predictor,
        )
        
        logger.info(f"[GlobalScheduler] Added replica {rid} to scheduler (total replicas: {self._num_replicas})")
        return rid

    # -------------------- Optional stats --------------------
    def get_migration_stats(self) -> dict:
        return {
            "total_migrations": self._migration_count,
            "cluster_freeness": {rid: F for rid, F in self._all_freeness()},
            "imbalance_gap": self._imbalance_gap(),
            "autoscale": self.autoscale_recommendation(),
        }
    

    def step(self):
        """
        Llumnix global scheduler step:
        1. Place new requests onto replicas (schedule())
        2. Ask each replica for a batch via _get_next_batch()
        3. Emit BatchStageArrivalEvent for each batch (advances replicas' stage schedulers)
        """
        from vidur.events.batch_stage_arrival_event import BatchStageArrivalEvent
        
        events = []

        # 1. Place any remaining global requests
        assignments = self.schedule()
        # (assignments are ignored for event creation; llumlets handle enqueue_request)

        # 2. Ask each replica for a batch
        for rid, sched in self._replica_schedulers.items():
            batch = sched._get_next_batch()
            
            # 3. Emit BatchStageArrivalEvent (tells stage schedulers to process this batch)
            # stage_id is 0 for single-stage replicas, or incremental for multi-stage
            if batch:
                events.append(
                    BatchStageArrivalEvent(self.current_time, rid, 0, batch)
                )

        return events



