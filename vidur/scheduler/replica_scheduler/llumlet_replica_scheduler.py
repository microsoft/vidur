from typing import List, Optional, Tuple, Dict, Any
import math

from vidur.entities import Request, Batch
from vidur.scheduler.replica_scheduler.base_replica_scheduler import BaseReplicaScheduler
from vidur.logger import init_logger

logger = init_logger(__name__)


class LlumletLocalScheduler(BaseReplicaScheduler):
    """
    Llumnix 'llumlet' — per-replica local scheduler with policy-faithful freeness.

    Freeness F = (M - ΣV) / B, where:
      - M: total KV blocks on the replica.
      - ΣV: virtual usage sum from multiple sources:
          * Physical KV in-use by running requests.
          * Head-of-line (HoL) queued *demand* in KV blocks (de-frag pressure).
          * Execution-priority headroom for high-priority requests.
          * Optional drain pressure (fake ∞) when replica is marked draining.
      - B: batch-normalization denominator (blocks per batch); defaults to 1.
    """

    # -------------------- Construction --------------------
    def __init__(
        self,
        replica_config,
        replica_scheduler_config,
        request_generator_config,
        replica,
        num_stages,
        execution_time_predictor,
    ):
        # BaseReplicaScheduler sets:
        #   self._config = replica_scheduler_config
        #   self._replica_id = replica.id
        #   self._num_stages
        #   self._allocation_map, self._num_allocated_blocks
        #   self._replica_stage_schedulers (for Chrome trace + stages)
        super().__init__(
            replica_config,
            replica_scheduler_config,
            request_generator_config,
            replica,
            num_stages,
            execution_time_predictor,
        )

        # Queue holds (priority, monotonic_seq, Request)
        # Lower priority value = "higher" priority (0 > 1 > 2...)
        self._priority_queue: List[Tuple[int, int, Request]] = []
        self._enqueue_seq: int = 0

        # Track number of running batches (BaseReplicaScheduler
        # also uses this; we only DECREMENT it in on_batch_end).
        self._num_running_batches: int = 0

        # Book-keeping for migration and priority-aware logic
        self._request_index: Dict[int, Request] = {}
        # Reservations local to this replica (used when this replica is the *destination*)
        self._reservations: Dict[int, int] = {}  # req_id -> reserved_blocks

        # Multi-stage migration state for outgoing migrations:
        # req_id -> {
        #   "dest": LlumletLocalScheduler,
        #   "blocks": int,
        #   "stages_total": int,
        #   "stages_done": int,
        # }
        self._migrations_out: Dict[int, Dict[str, Any]] = {}

        # Optional drain flag (for scale-in)
        self._is_draining: bool = False

        # Tunables from replica scheduler config (stored in self._config)
        cfg = self._config
        
        # Per Algorithm 1 line 10: each priority has its own headroom budget
        # Get num_priority_levels from request_generator_config
        self._num_priority_levels: int = getattr(request_generator_config, "num_priority_levels", 5)
        
        # Per-priority headroom pools (Algorithm 1, line 10: headroomForPriority[p])
        # Headroom should be scaled relative to replica capacity (M = num_blocks)
        # Paper uses headroom as fraction of capacity, not absolute value
        M = max(1, cfg.num_blocks)
        
        # Get headroom decay mode from config
        headroom_decay_mode = getattr(cfg, "headroom_decay_mode", "exponential")
        
        # Calculate headroom for each priority level using configurable decay
        self._headroom_for_priority: List[int] = self._calculate_headroom_distribution(
            M, self._num_priority_levels, headroom_decay_mode
        )
        
        # NOTE: Batch normalizer B is now DYNAMIC (current batch size = len(allocation_map))
        # Previously used static config value, now computed on-demand in report_freeness()
        # per paper Section 4.4.1: "batch size determines consumption speed"
        
        # Migration stage granularity: how many KV blocks per migration stage
        self._migration_stage_blocks: int = getattr(cfg, "migration_stage_blocks", 1) or 1
    
    def _calculate_headroom_distribution(
        self, capacity: int, num_levels: int, decay_mode: str
    ) -> List[int]:
        """
        Calculate headroom for each priority level with configurable decay.
        
        Args:
            capacity: Total KV cache capacity (M = num_blocks)
            num_levels: Number of priority levels (1-10 supported)
            decay_mode: "linear" or "exponential"
        
        Returns:
            List of headroom values (in blocks) for each priority level
        
        Headroom semantics:
        - Priority 0 (highest): Gets largest headroom
        - Priority N-1 (lowest): Gets minimal/no headroom
        - Decay controls how quickly headroom decreases
        """
        if num_levels <= 0:
            return []
        
        if num_levels == 1:
            # Single priority: no differentiation needed
            return [0]
        
        headroom = []
        
        if decay_mode == "linear":
            # Linear decay: evenly spaced from max to min
            # Highest priority gets 20% of capacity
            # Lowest priority gets 0%
            max_headroom_fraction = 0.20
            for p in range(num_levels):
                # Linear interpolation from max to 0
                fraction = max_headroom_fraction * (1.0 - p / (num_levels - 1))
                headroom.append(int(capacity * fraction))
        
        elif decay_mode == "exponential":
            # Exponential decay: rapid decrease for lower priorities
            # Highest priority gets 20% of capacity
            # Decay factor ensures smooth exponential curve
            import math
            max_headroom_fraction = 0.20
            
            # Decay constant tuned for num_levels
            # Ensures lowest priority gets near-zero headroom
            decay_constant = 2.5 / max(1, num_levels - 1)
            
            for p in range(num_levels):
                # Exponential decay: e^(-decay_constant * p)
                fraction = max_headroom_fraction * math.exp(-decay_constant * p)
                headroom.append(int(capacity * fraction))
        
        else:
            # Default to exponential if unknown mode
            import math
            max_headroom_fraction = 0.20
            decay_constant = 2.5 / max(1, num_levels - 1)
            for p in range(num_levels):
                fraction = max_headroom_fraction * math.exp(-decay_constant * p)
                headroom.append(int(capacity * fraction))
        
        return headroom

    # -------------------- Properties --------------------
    @property
    def num_pending_requests(self) -> int:
        """Override to use our priority queue instead of base _request_queue."""
        return len(self._priority_queue)

    # -------------------- Queueing & batching --------------------
    def add_request(self, request: Request) -> None:
        """
        Override base class to use priority queue instead of simple queue.
        This prevents duplicate queueing when GlobalScheduleEvent calls add_request().
        """
        self.enqueue_request(request)
    
    def enqueue_request(self, request: Request) -> None:
        """
        Insert request into priority queue.
        """
        pr = getattr(request, "priority", 0)
        self._enqueue_seq += 1
        self._priority_queue.append((pr, self._enqueue_seq, request))
        # sort by (priority, seq): smaller priority first, then FIFO
        self._priority_queue.sort(key=lambda x: (x[0], x[1]))
        self._request_index[request.id] = request

    def _pop_next_request(self) -> Optional[Request]:
        if not self._priority_queue:
            return None
        _, _, req = self._priority_queue.pop(0)
        return req

    # Override free to be tolerant (pop(..., None))
    def free(self, *request_ids: int) -> None:
        for request_id in request_ids:
            num_blocks = self._allocation_map.pop(request_id, None)
            if num_blocks is not None:
                self._num_allocated_blocks -= num_blocks
        assert self._num_allocated_blocks >= 0

    def _peek_hol_request(self) -> Optional[Request]:
        if not self._priority_queue:
            return None
        return self._priority_queue[0][2]

    def _blocks_for_request_next_step(self, req: Request) -> int:
        # Prefill: allocate KV footprint
        if not req.is_prefill_complete:
            tokens = req.num_prefill_tokens
            block = getattr(self._config, "block_size", 1)
            return max(1, math.ceil(tokens / block))

        # Decode: no new KV blocks needed
        return 0

    def _get_next_batch(self) -> Optional[Batch]:
        """
        Build a continuous-batching multi-request batch:
        • Greedily pick requests in priority/FIFO order
        • Include as many as fit in remaining KV memory
        • Stop when next request cannot be allocated
        • Remove chosen requests from the queue
        • Allocate KV for each request

        NOTE: DO NOT touch self._num_running_batches here.
        BaseReplicaScheduler.on_schedule() increments it
        once per returned batch. We only decrement in on_batch_end.
        """

        if not self._priority_queue:
            return None

        chosen_entries: List[Tuple[int, int, Request]] = []
        chosen_requests: List[Request] = []
        total_blocks = 0
        remaining_queue = list(self._priority_queue)  # snapshot

        for pr, seq, req in remaining_queue:
            blocks = self._blocks_for_request_next_step(req)

            # If adding this request exceeds KV capacity → stop packing
            if not self.can_allocate(total_blocks + blocks):
                break

            total_blocks += blocks
            chosen_entries.append((pr, seq, req))
            chosen_requests.append(req)

        if not chosen_requests:
            return None

        # Remove selected from real queue
        for item in chosen_entries:
            self._priority_queue.remove(item)

        # Allocate KV blocks and keep index
        for req in chosen_requests:
            blocks = self._blocks_for_request_next_step(req)
            self.allocate(req.id, blocks)
            self._request_index[req.id] = req

        # Per-request token counts for this iteration
        num_tokens = [self._get_request_next_num_tokens(req) for req in chosen_requests]

        # IMPORTANT: DO NOT increment _num_running_batches here
        # BaseReplicaScheduler.on_schedule() will do that.

        return Batch(
            replica_id=self._replica_id,
            requests=chosen_requests,
            num_tokens=num_tokens,
        )

    def get_batch_compute_cost(self, batch: Batch) -> float:
        """
        Return simulated compute duration (in seconds) to process this batch.
        vLLM/Llumnix semantics:
        - One decode iteration per batch
        - Cost grows with the number of active requests (batch size)
        """
        base_cost = getattr(self._config, "per_request_compute_cost", 0.01)
        batch_size = len(batch.requests)
        return base_cost * batch_size

    def on_batch_end(self, batch: Batch) -> None:
        """
        Called after Batch.on_batch_end has updated Request objects.

        CRITICAL FIX: Properly manage KV allocation lifecycle during migration.
        """
        # 1) Decrement running batch counter FIRST
        self._num_running_batches = max(0, self._num_running_batches - 1)

        # 2) Process each request in the batch
        for req in batch.requests:
            req_id = req.id
            mig_state = self._migrations_out.get(req_id)

            # Get current allocation (will be freed at end of this block)
            current_blocks = self._allocation_map.get(req_id, 0)

            # Requests with migration state (multi-stage live migration)
            if mig_state is not None:
                dest_sched: "LlumletLocalScheduler" = mig_state["dest"]
                blocks: int = mig_state["blocks"]
                stages_total: int = mig_state["stages_total"]
                stages_done: int = mig_state["stages_done"]

                if req.completed:
                    # Request finished before migration completed → abort.
                    self.free(req_id)  # FREE FIRST
                    dest_sched._abort_reservation(req_id)
                    self._migrations_out.pop(req_id, None)
                    self._request_index.pop(req_id, None)
                    logger.info(
                        f"[Migration] Request {req_id} completed on source replica "
                        f"{self.replica_id} before migration finished; aborted."
                    )
                    continue

                # Advance migration stage
                stages_done += 1
                mig_state["stages_done"] = stages_done

                if stages_done < stages_total:
                    # Still copying KV; free current batch allocation, re-enqueue
                    self.free(req_id)  # FREE BEFORE RE-ENQUEUE
                    
                    pr = getattr(req, "priority", 0)
                    self._enqueue_seq += 1
                    self._priority_queue.append((pr, self._enqueue_seq, req))
                    self._priority_queue.sort(key=lambda x: (x[0], x[1]))
                    logger.debug(
                        f"[Replica {self._replica_id}] Request {req_id} migration stage "
                        f"{stages_done}/{stages_total}; freed & re-enqueued."
                    )
                    continue

                # Final stage: attempt commit on destination
                if dest_sched._dest_commit_if_reserved(req_id, blocks):
                    # SUCCESS: free on source, remove tracking
                    self.free(req_id)
                    self._migrations_out.pop(req_id, None)
                    self._request_index.pop(req_id, None)

                    # Remove from queue if somehow still there
                    self._priority_queue = [
                        (pr, seq, r) for (pr, seq, r) in self._priority_queue
                        if r.id != req_id
                    ]

                    # Add to destination
                    dest_sched._request_index[req_id] = req
                    dest_sched.enqueue_request(req)

                    logger.info(
                        f"[Migration] Request {req_id} completed migration "
                        f"{self.replica_id} -> {dest_sched.replica_id}"
                    )
                else:
                    # FAILED: abort, free blocks, keep on source
                    self.free(req_id)  # FREE BEFORE RE-ENQUEUE
                    dest_sched._abort_reservation(req_id)
                    self._migrations_out.pop(req_id, None)

                    pr = getattr(req, "priority", 0)
                    self._enqueue_seq += 1
                    self._priority_queue.append((pr, self._enqueue_seq, req))
                    self._priority_queue.sort(key=lambda x: (x[0], x[1]))
                    logger.info(
                        f"[Migration] Request {req_id} migration aborted at final stage; "
                        f"freed & kept on source {self.replica_id}"
                    )

                continue  # done with migrating request

            # Non-migrating requests
            if req.completed:
                self.free(req_id)  # FREE COMPLETED REQUEST
                self._request_index.pop(req_id, None)
                self._reservations.pop(req_id, None)
                logger.debug(f"[Replica {self._replica_id}] Request {req_id} completed & freed.")
            else:
                # FREE BEFORE RE-ENQUEUE
                self.free(req_id)
                
                pr = getattr(req, "priority", 0)
                self._enqueue_seq += 1
                self._priority_queue.append((pr, self._enqueue_seq, req))
                self._priority_queue.sort(key=lambda x: (x[0], x[1]))
                logger.debug(
                    f"[Replica {self._replica_id}] Request {req_id} freed & re-enqueued."
                )

    # -------------------- Virtual-usage policy --------------------
    def _virtual_usage_physical(self) -> int:
        return int(self._num_allocated_blocks)

    def _virtual_usage_hol_demand(self) -> int:
        hol = self._peek_hol_request()
        if not hol:
            return 0
        return self._blocks_for_request_next_step(hol)

    def _virtual_usage_priority_headroom(self) -> int:
        """
        Calculate total headroom contribution per Algorithm 1, lines 8-10:
        
        Line 8: virtualUsage = physicalUsage + GetHeadroom(priority, instance)
        Line 10: GetHeadroom(p, instance) = headroomForPriority[p] / instance.numRequests[p]
        
        Each priority level has its own headroom budget.
        Total virtual usage from headroom = sum of each priority's full headroom budget
        (if any requests exist at that priority).
        """
        total_headroom = 0
        
        # Count requests at each priority level
        requests_per_priority = [0] * self._num_priority_levels
        
        # Count queued requests
        for pr, _, _req in self._priority_queue:
            if 0 <= pr < self._num_priority_levels:
                requests_per_priority[pr] += 1
        
        # Count running requests
        for rid in list(self._allocation_map.keys()):
            req = self._request_index.get(rid)
            if req:
                pr = getattr(req, "priority", 0)
                if 0 <= pr < self._num_priority_levels:
                    requests_per_priority[pr] += 1
        
        # Per Algorithm 1 line 10: for each priority with requests,
        # headroom contribution = headroomForPriority[p] / numRequests[p] * numRequests[p]
        # This simplifies to: if numRequests[p] > 0, add full headroomForPriority[p]
        for priority in range(self._num_priority_levels):
            if requests_per_priority[priority] > 0 and self._headroom_for_priority[priority] > 0:
                total_headroom += self._headroom_for_priority[priority]
        
        return total_headroom

    def _virtual_usage_drain(self) -> int:
        if not self._is_draining:
            return 0
        return 10 * max(1, self._config.num_blocks)

    def _sum_virtual_usage(self) -> int:
        return (
            self._virtual_usage_physical()
            + self._virtual_usage_hol_demand()
            + self._virtual_usage_priority_headroom()
            + self._virtual_usage_drain()
        )

    def report_freeness(self) -> float:
        """
        Llumnix freeness metric: F = (M - ΣV) / B
        
        Per paper Section 4.4.1: "We divide it by the batch size because it 
        determines the consumption speed, i.e., the number of new tokens per 
        iteration. Thus the metric suggests how many iterations the batch can 
        still run for."
        
        B = current batch size (number of running requests), NOT static config.
        Each running request consumes ~1 block per decode iteration.
        """
        M = max(1, self._config.num_blocks)
        SigmaV = self._sum_virtual_usage()
        # B = DYNAMIC batch size (consumption rate in blocks/iteration)
        B = max(1, len(self._allocation_map))
        return (M - SigmaV) / B  # negative allowed

    def report_normal_priority_freeness(self) -> float:
        """
        Calculate freeness considering only normal-priority requests.
        Per paper Section 4.4.3: autoscaling uses "average freeness for the normal priority".
        
        This excludes priority headroom from high-priority requests to avoid
        over-provisioning the cluster due to virtual usage inflation.
        
        B = current batch size (dynamic), not static config value.
        """
        M = max(1, self._config.num_blocks)
        # Sum virtual usage WITHOUT priority headroom
        physical = self._virtual_usage_physical()
        hol = self._virtual_usage_hol_demand()
        drain = self._virtual_usage_drain()
        SigmaV = physical + hol + drain
        # B = DYNAMIC batch size (consumption rate in blocks/iteration)
        B = max(1, len(self._allocation_map))
        
        freeness = (M - SigmaV) / B
        
        # Debug log when freeness is constant/negative (potential overload)
        if freeness < -100 and self._replica_id == 0:  # Only log from replica 0 to avoid spam
            logger.debug(
                f"[Replica {self._replica_id}] Normal freeness={freeness:.1f}: "
                f"M={M}, physical={physical}, hol={hol}, drain={drain}, "
                f"SigmaV={SigmaV}, B={B}, "
                f"queue_len={len(self._priority_queue)}, "
                f"alloc_size={len(self._allocation_map)}"
            )
        
        return freeness

    def has_capacity(self, num_blocks: int = 1) -> bool:
        return self.can_allocate(num_blocks)

    # -------------------- Migration handshake --------------------
    def _reserve_on_dest(self, dest: "LlumletLocalScheduler", req_id: int, blocks: int) -> bool:
        if dest._reservations.get(req_id):
            return True
        if not dest.can_allocate(blocks):
            return False
        dest._reservations[req_id] = blocks
        return True

    def _abort_reservation(self, req_id: int) -> None:
        self._reservations.pop(req_id, None)

    def _dest_commit_if_reserved(self, req_id: int, blocks: int) -> bool:
        if self._reservations.get(req_id) != blocks:
            return False
        if not self.can_allocate(blocks):
            self._reservations.pop(req_id, None)
            return False
        self.allocate(req_id, blocks)
        self._reservations.pop(req_id, None)
        logger.info(
            f"[Migration] Request {req_id} successfully committed on destination "
            f"replica {self.replica_id} ({blocks} blocks)"
        )
        return True

    def decide_migration_candidate(self, target_capacity_blocks: int) -> Optional[int]:
        """
        Llumnix migration candidate selection (Algorithm 1 logic):
        Prefer low priority (higher priority value) and shorter sequence lengths.
        
        When draining (auto-scaling): Prioritize RUNNING requests over queued ones
        to migrate actual KV state. This matches the paper's intent to drain replicas
        by migrating "remaining requests" with their allocations.
        """
        running_candidates: List[Tuple[int, int, int]] = []  # (priority, blocks, req_id)
        queued_candidates: List[Tuple[int, int, int]] = []

        # Running requests — these have actual KV blocks allocated
        for req_id, blocks in self._allocation_map.items():
            if blocks <= target_capacity_blocks:
                req = self._request_index.get(req_id)
                pr = getattr(req, "priority", 0)
                running_candidates.append((pr, blocks, req_id))

        # Queued requests — approximate blocks using next-step demand
        for pr, _, req in self._priority_queue:
            b = self._blocks_for_request_next_step(req)
            if b <= target_capacity_blocks:
                queued_candidates.append((pr, b, req.id))

        # When draining, prioritize running requests (they have actual state to migrate)
        # Otherwise, use normal load-balancing (which may include queued for fairness)
        if self._is_draining and running_candidates:
            candidates = running_candidates
        else:
            candidates = running_candidates + queued_candidates

        if not candidates:
            return None
        
        # Llumnix preference: lower priorities (higher priority value) first, then smallest KV
        # This matches the paper's goal of migrating low-priority, short-sequence requests
        candidates.sort(key=lambda t: (-t[0], t[1]))
        return candidates[0][2]

    def begin_migration_to(self, dest_scheduler: "LlumletLocalScheduler") -> Optional[Tuple[int, int, int]]:
        """
        Llumnix-style multi-stage live migration:

        1) Choose candidate (low-pri, small KV).
        2) If queued: cold-migrate immediately (no staging).
        3) If running: set up multi-stage migration state; each batch advances one stage.
        
        CRITICAL: When source replica is draining (auto-scaling), prioritize migrating
        ALL requests, not just load-balancing. The global scheduler will have marked
        the source as draining via set_draining(True), triggering this aggressive behavior.
        """

        # PRE-FLIGHT VALIDATION
        def request_exists(req_id: int) -> bool:
            return (
                req_id in self._request_index
                or req_id in self._allocation_map
                or any(_r.id == req_id for (_p, _s, _r) in self._priority_queue)
            )

        # Compute free space on dest (including reservations)
        dest_free = dest_scheduler._config.num_blocks - (
            dest_scheduler._num_allocated_blocks +
            sum(dest_scheduler._reservations.values())
        )
        if dest_free <= 0:
            return None

        # Pick migration candidate
        # When draining, prioritize migrating ALL requests; otherwise use smart selection
        cand_id = self.decide_migration_candidate(dest_free)
        if cand_id is None:
            return None

        if not request_exists(cand_id):
            logger.warning(
                f"[Migration] Skipped migration of req {cand_id}: request no longer exists "
                f"on replica {self.replica_id}"
            )
            return None

        # Determine whether running or queued
        blocks = self._allocation_map.get(cand_id)

        # QUEUED REQUEST MIGRATION (cold)
        if blocks is None:
            req = self._request_index.get(cand_id)
            if not req:
                logger.warning(
                    f"[Migration] Skipped: request {cand_id} vanished before migration "
                    f"from replica {self.replica_id}"
                )
                return None

            blocks = self._blocks_for_request_next_step(req)

            # Remove from queue
            removed = False
            for i, (_pr, _seq, _r) in enumerate(list(self._priority_queue)):
                if _r.id == cand_id:
                    self._priority_queue.pop(i)
                    removed = True
                    break

            if not removed:
                logger.warning(
                    f"[Migration] Request {cand_id} not found in queue during migration "
                    f"on replica {self.replica_id} — skipping"
                )
                return None

            # Push into destination queue
            dest_scheduler.enqueue_request(req)

            log_level = "info" if self._is_draining else "debug"
            getattr(logger, log_level)(
                f"[Migration] Queued req {cand_id} cold-migrated from replica {self.replica_id} "
                f"-> {dest_scheduler.replica_id} (blocks={blocks}){' [DRAINING]' if self._is_draining else ''}"
            )

            return (cand_id, self.replica_id, dest_scheduler.replica_id)

        # ----------------------------
        # RUNNING REQUEST MIGRATION (multi-stage)
        req = self._request_index.get(cand_id)
        if not req:
            return None

        # If already migrating, don't double-start
        if cand_id in self._migrations_out:
            return None

        # Reserve full KV footprint on destination
        if not self._reserve_on_dest(dest_scheduler, cand_id, blocks):
            return None

        stages_total = max(1, math.ceil(blocks / self._migration_stage_blocks))

        self._migrations_out[cand_id] = {
            "dest": dest_scheduler,
            "blocks": blocks,
            "stages_total": stages_total,
            "stages_done": 0,
        }

        log_level = "info" if self._is_draining else "debug"
        getattr(logger, log_level)(
            f"[Migration] Running req {cand_id} scheduled for multi-stage migration "
            f"{self.replica_id} -> {dest_scheduler.replica_id} "
            f"(blocks={blocks}, stages={stages_total}){' [DRAINING]' if self._is_draining else ''}"
        )

        return (cand_id, self.replica_id, dest_scheduler.replica_id)

    def set_draining(self, draining: bool) -> None:
        self._is_draining = draining

    def is_empty(self) -> bool:
        """
        Replica is empty when:
          • no queued requests
          • no running/allocated requests
          • no in-flight migrations
        """
        return (
            len(self._priority_queue) == 0
            and len(self._allocation_map) == 0
            and len(self._migrations_out) == 0
        )

    def _compute_temperature(self) -> float:
        """
        Returns virtual usage as a temperature (can exceed 1.0 when overloaded).
        
        INTENT: Show the actual freeness level that drives migration and auto-scaling.
        When virtual usage is high (red), migrations and draining are triggered.
        When virtual usage is low (green), system is balanced and ready for consolidation.
        
        This reveals the true system state metric that drives all scheduling decisions.
        """
        M = max(1, self._config.num_blocks)
        virtual_usage = self._sum_virtual_usage()
        
        # Raw metric: virtual usage as multiple of capacity
        # Can exceed 1.0 to show overload intensity
        temperature = virtual_usage / M
        
        return temperature

    def _temperature_color(self) -> str:
        """
        Chrome trace color bucket based on temperature (freeness metric).
        Maps temperature → Chrome trace *reserved* color name.
        
        THRESHOLDS (adjusted for 20-10-5% headroom baseline):
        With current headroom settings (20%+10%+5% ≈ 34% of capacity reserved),
        even idle replicas have baseline temperature ~0.34.
        
        Adjusted thresholds to show meaningful color progression:
        - < 0.40 (40%):  good (green) - idle/very light load
        - < 0.55 (55%):  rail_idle (light green) - light load
        - < 0.70 (70%):  rail_animation (yellow) - moderate load
        - < 0.85 (85%):  terrible (orange) - high load, nearing capacity
        - >= 0.85:       bad (red) - very high load / overloaded (>85%)
        
        These thresholds show when migration/draining would be triggered.
        """
        t = self._compute_temperature()

        # Must use only allowed Chrome names
        # Adjusted thresholds for 20-10-5% headroom (baseline ~34% of capacity)
        # Even idle replicas have temperature ~0.34 due to headroom reservation
        if t < 0.40:
            return "good"              # green (idle/very light, below 40%)
        elif t < 0.55:
            return "rail_idle"         # light green (light load, 40-55%)
        elif t < 0.70:
            return "rail_animation"    # yellow (moderate load, 55-70%)
        elif t < 0.85:
            return "terrible"          # orange (high load, 70-85%)
        else:
            return "bad"               # red (very high/overloaded, 85%+)

    def _emit_chrome_trace_batch(self, batch: Batch, start_time: float, end_time: float) -> None:
        """
        Override Vidur's default batch trace emit to include:
        - KV virtual usage visualization
        - Temperature color
        """
        temperature = self._compute_temperature()

        args = {
            "temperature": temperature,
            "virtual_usage": self._sum_virtual_usage(),
            "physical_usage": self._virtual_usage_physical(),
            "hol_demand": self._virtual_usage_hol_demand(),
            "priority_headroom": self._virtual_usage_priority_headroom(),
            "num_requests": len(batch.requests),
        }

        self._emit_trace_event(
            name="batch",
            category="schedule",
            start_time=start_time,
            end_time=end_time,
            cname=self._temperature_color(),   # COLOR GOES HERE
            args=args,
        )
