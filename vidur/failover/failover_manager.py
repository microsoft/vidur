"""Replica failure injection and recovery.

Implements the recovery strategies compared in LUMEN (Cao et al., "LUMEN:
Coordinated Failure Recovery for Distributed LLM Serving", arXiv:2606.17787):

* stop_and_restart: the failed replica reloads; its interrupted requests are
  re-dispatched to surviving replicas and recomputed from scratch.
* fixed_checkpointing: every request streams its KV cache to a fixed
  neighbouring replica (its checkpoint holder, as in DéjàVu) and resumes there.
* lumen: (i) load-aware KV checkpointing, (ii) locality-aware recovery
  scheduling and (iii) speculation-assisted progressive recovery.

A LUMEN "worker" is a replica here, so a 4-stage pipeline fails as a whole.
"""

import json
import os
import random
from collections import defaultdict
from math import ceil
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from vidur.config import SimulationConfig
from vidur.entities import Batch, Request
from vidur.failover.failure_record import FailureRecord, InterruptedRequest
from vidur.failover.kv_checkpoint_tracker import KVCheckpointTracker
from vidur.logger import init_logger
from vidur.types import FailureRecoveryPolicyType, ReplicaSchedulerType
from vidur.utils.param_counter import ParamCounter

logger = init_logger(__name__)

GB = 1e9


def summarize_latencies(requests_df: pd.DataFrame) -> dict:
    summary = {"num_requests": len(requests_df)}
    for metric in ["ttft", "tpot", "e2e_time"]:
        values = requests_df[metric].to_numpy()
        summary[metric] = (
            {
                "mean": float(np.mean(values)),
                "p50": float(np.percentile(values, 50)),
                "p90": float(np.percentile(values, 90)),
                "p99": float(np.percentile(values, 99)),
            }
            if len(values)
            else None
        )
    return summary


class FailoverManager:
    def __init__(self, config: SimulationConfig, scheduler) -> None:
        self._config = config.failure_config
        self._output_dir = config.metrics_config.output_dir
        self._scheduler = scheduler
        self._policy = self._config.recovery_policy_type
        self._enabled = self._policy != FailureRecoveryPolicyType.NONE

        assert self._enabled or not self._config.failure_times, (
            "Failures were requested without a recovery policy, "
            "set --failure_config_recovery_policy"
        )

        replica_config = config.cluster_config.replica_config
        model_config = replica_config.model_config
        self._num_replicas = config.cluster_config.num_replicas

        if self._enabled:
            scheduler_type = config.cluster_config.replica_scheduler_config.get_type()
            assert scheduler_type in (
                ReplicaSchedulerType.SARATHI,
                ReplicaSchedulerType.VLLM,
            ), "Failure recovery supports the sarathi and vllm replica schedulers"
            for replica_id in range(self._num_replicas):
                assert replica_id in scheduler._replica_schedulers
            for replica_id in self._config.failure_replica_ids:
                assert 0 <= replica_id < self._num_replicas

        is_lumen = self._policy == FailureRecoveryPolicyType.LUMEN
        self._checkpointing = self._policy in (
            FailureRecoveryPolicyType.FIXED_CHECKPOINTING,
            FailureRecoveryPolicyType.LUMEN,
        )
        self._load_aware_checkpointing = (
            is_lumen and self._config.lumen_load_aware_checkpointing
        )
        self._locality_aware_scheduling = (
            is_lumen and self._config.lumen_locality_aware_scheduling
        )
        self._speculative_recovery = (
            is_lumen and self._config.lumen_speculative_recovery
        )

        # K and V in fp16 for every layer of the model
        head_dim = model_config.embedding_dim // model_config.num_q_heads
        self._kv_bytes_per_token = (
            2 * 2 * head_dim * model_config.num_kv_heads * model_config.num_layers
        )
        # each device restores its own layers and heads in parallel
        kv_heads_per_device = ceil(
            model_config.num_kv_heads / replica_config.tensor_parallel_size
        )
        kv_bytes_per_token_per_device = (
            2
            * 2
            * head_dim
            * kv_heads_per_device
            * (model_config.num_layers // replica_config.num_pipeline_stages)
        )
        self._restore_seconds_per_token = kv_bytes_per_token_per_device / (
            self._config.restore_bandwidth_gb_per_s_per_device * GB
        )

        weight_bytes_per_device = (
            2 * ParamCounter(replica_config).get_num_parameters_per_device()
        )
        if self._config.model_reload_time_s is not None:
            self._model_reload_time = self._config.model_reload_time_s
        else:
            self._model_reload_time = (
                self._config.engine_restart_overhead_s
                + weight_bytes_per_device
                / (self._config.weight_load_bandwidth_gb_per_s * GB)
            )

        self._checkpoints = KVCheckpointTracker(
            self._num_replicas,
            self._kv_bytes_per_token,
            self._config.checkpoint_bandwidth_gb_per_s * GB,
            self._config.checkpoint_granularity_tokens,
        )

        # which replica each outstanding request is assigned to
        self._replica_requests: Dict[int, Dict[int, Request]] = {
            replica_id: {} for replica_id in range(self._num_replicas)
        }
        self._request_replica: Dict[int, int] = {}
        # host-to-device copies of restored checkpoints, per target replica
        self._restore_free_at = [0.0] * self._num_replicas
        self._num_restoring_tokens = [0] * self._num_replicas

        self._failures: List[FailureRecord] = []
        self._active_failures: Dict[int, FailureRecord] = {}
        self._awaiting_resume: Dict[int, List[InterruptedRequest]] = defaultdict(list)

        # draft model capacity offered by recovering replicas
        self._num_speculative_slots = 0
        self._speculative_requests: Dict[int, Request] = {}
        self._num_verification_steps = 0
        self._num_draft_tokens = 0
        self._num_accepted_draft_tokens = 0
        self._rng = random.Random(config.seed)

        self._output_token_watermark: Dict[int, int] = {}
        self._num_output_tokens_per_second: Dict[int, int] = defaultdict(int)
        self._completed_requests: List[dict] = []
        self._interrupted_request_ids: Set[int] = set()

        if self._enabled:
            logger.info(
                f"Failure recovery policy {self._policy}: KV cache "
                f"{self._kv_bytes_per_token / 2**20:.2f} MiB/token, "
                f"model reload {self._model_reload_time:.1f}s, "
                f"failures at {self._config.failure_times}"
            )

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def model_reload_time(self) -> float:
        return self._model_reload_time

    def get_initial_events(self) -> list:
        from vidur.events.replica_failure_event import ReplicaFailureEvent

        return [
            ReplicaFailureEvent(time, replica_id, stage_id)
            for time, replica_id, stage_id in zip(
                self._config.failure_times,
                self._config.failure_replica_ids,
                self._config.failure_stage_ids,
            )
        ]

    # ------------------------------------------------------------------
    # bookkeeping during normal serving
    # ------------------------------------------------------------------

    def on_request_routed(self, replica_id: int, request: Request) -> None:
        if not self._enabled:
            return

        self._assign(request, replica_id)

    def on_before_batch_end(self, batch: Batch) -> None:
        """Decide how many speculative tokens each verification accepts."""
        if not self._speculative_recovery:
            return

        num_committed_tokens = None
        for i, (request, num_tokens) in enumerate(
            zip(batch.requests, batch.num_tokens)
        ):
            # a decode iteration that verifies draft tokens
            if not request.is_prefill_complete or num_tokens == 1:
                continue

            if num_committed_tokens is None:
                num_committed_tokens = list(batch.num_tokens)

            num_draft_tokens = num_tokens - 1
            num_accepted = 0
            while (
                num_accepted < num_draft_tokens
                and self._rng.random() < self._config.speculative_acceptance_rate
            ):
                num_accepted += 1

            # the target model always contributes one token of its own
            num_committed_tokens[i] = num_accepted + 1
            self._num_verification_steps += 1
            self._num_draft_tokens += num_draft_tokens
            self._num_accepted_draft_tokens += num_accepted

        if num_committed_tokens is not None:
            batch.num_committed_tokens = num_committed_tokens

    def on_batch_end(self, time: float, replica_id: int, batch: Batch) -> list:
        if not self._enabled:
            return []

        for request in batch.requests:
            self._record_output_tokens(time, request)
            self._check_resumed(time, request)

            if request.completed:
                self._on_request_completed(request)
            elif self._checkpointing:
                self._stream_checkpoint(time, replica_id, request)

        return self._continue_speculation(time, replica_id, batch)

    # ------------------------------------------------------------------
    # failure lifecycle
    # ------------------------------------------------------------------

    def on_replica_failure(
        self, time: float, replica_id: int, stage_id: int, metrics_store
    ) -> list:
        from vidur.events.failure_detected_event import FailureDetectedEvent

        replica_scheduler = self._scheduler.get_replica_scheduler(replica_id)
        if not replica_scheduler.is_alive:
            logger.warning(
                f"Ignoring failure of replica {replica_id} at {time}s: it is already down"
            )
            return []

        logger.info(f"Replica {replica_id} (stage {stage_id}) failed at {time:.3f}s")

        record = FailureRecord(replica_id, stage_id, time)
        self._failures.append(record)
        self._active_failures[replica_id] = record

        for request in self._replica_requests[replica_id].values():
            checkpoint = self._checkpoints.get(request.id)
            num_checkpointed_tokens = 0
            holder_replica_id = None
            if checkpoint and checkpoint.source_replica_id == replica_id:
                holder_replica_id = checkpoint.holder_replica_id
                if self._is_alive(holder_replica_id):
                    num_checkpointed_tokens = min(
                        checkpoint.get_num_durable_tokens(time),
                        request.num_processed_tokens,
                    )
            record.snapshots[request.id] = (
                request.num_processed_tokens,
                num_checkpointed_tokens,
                holder_replica_id,
            )
            self._stop_speculating(request)

        # stages that were mid-execution never finish
        for (
            stage_id_,
            stage_scheduler,
        ) in replica_scheduler._replica_stage_schedulers.items():
            if stage_scheduler.is_busy:
                metrics_store.on_batch_stage_end(None, time, replica_id, stage_id_)

        replica_scheduler.fail()
        metrics_store.on_replica_schedule(time, replica_id, 0)

        self._restore_free_at[replica_id] = time
        self._num_restoring_tokens[replica_id] = 0
        self._checkpoints.on_source_failed(replica_id, time)
        if self._config.holder_failure_loses_checkpoints:
            self._checkpoints.on_holder_failed(replica_id)

        return [FailureDetectedEvent(time + self._config.detection_delay_s, replica_id)]

    def on_failure_detected(self, time: float, replica_id: int) -> list:
        from vidur.events.draft_model_ready_event import DraftModelReadyEvent
        from vidur.events.replica_recovered_event import ReplicaRecoveredEvent

        record = self._active_failures[replica_id]
        record.detected_at = time

        self._scheduler.set_replica_routable(replica_id, False)
        # also drops requests routed to the replica after it failed
        self._scheduler.get_replica_scheduler(replica_id).reset()

        affected = sorted(
            self._replica_requests[replica_id].values(),
            key=lambda request: (request.arrived_at, request.id),
        )
        for request in affected:
            self._unassign(request)

        events = [ReplicaRecoveredEvent(time + self._model_reload_time, replica_id)]
        if (
            self._speculative_recovery
            and self._config.draft_model_load_time_s < self._model_reload_time
        ):
            events.append(
                DraftModelReadyEvent(
                    time + self._config.draft_model_load_time_s, replica_id
                )
            )

        events.extend(self._recover_requests(time, record, affected))

        logger.info(
            f"Replica {replica_id} failure detected at {time:.3f}s: "
            f"{len(record.interrupted)} interrupted, {record.num_requeued} requeued, "
            f"reload until {time + self._model_reload_time:.3f}s"
        )
        return events

    def on_request_restore_end(
        self,
        time: float,
        replica_id: int,
        request: Request,
        epoch: int,
        num_reserved_tokens: int,
    ) -> list:
        from vidur.events.replica_schedule_event import ReplicaScheduleEvent

        replica_scheduler = self._scheduler.get_replica_scheduler(replica_id)
        if replica_scheduler.epoch != epoch:
            # the target failed meanwhile and has already taken care of the request
            return []

        self._num_restoring_tokens[replica_id] -= num_reserved_tokens
        replica_scheduler.add_request(request)
        return [ReplicaScheduleEvent(time, replica_id)]

    def on_draft_model_ready(self, time: float, replica_id: int) -> list:
        record = self._active_failures.get(replica_id)
        if record is None or record.detected_at is None:
            return []

        logger.info(f"Draft model ready on replica {replica_id} at {time:.3f}s")
        record.draft_ready_at = time
        self._num_speculative_slots += self._config.speculative_max_requests
        return []

    def on_speculative_draft_end(
        self, time: float, replica_id: int, requests: List[Request], epoch: int
    ) -> list:
        from vidur.events.replica_schedule_event import ReplicaScheduleEvent

        replica_scheduler = self._scheduler.get_replica_scheduler(replica_id)
        if replica_scheduler.epoch != epoch:
            return []

        for request in requests:
            if request.id in self._speculative_requests:
                request.num_speculative_tokens = (
                    self._config.speculative_num_draft_tokens
                )
            else:
                request.num_speculative_tokens = 0

        replica_scheduler.release_requests(requests)
        return [ReplicaScheduleEvent(time, replica_id)]

    def on_replica_recovered(self, time: float, replica_id: int) -> list:
        from vidur.events.global_schedule_event import GlobalScheduleEvent

        record = self._active_failures.pop(replica_id)
        record.recovered_at = time
        logger.info(f"Replica {replica_id} recovered at {time:.3f}s")

        self._scheduler.get_replica_scheduler(replica_id).recover()
        self._scheduler.set_replica_routable(replica_id, True)

        if record.draft_ready_at is not None:
            # the draft model makes way for the full model
            self._num_speculative_slots -= self._config.speculative_max_requests
            if self._num_speculative_slots == 0:
                for request in list(self._speculative_requests.values()):
                    self._stop_speculating(request)

        # requests may have queued up while no replica was routable
        return [GlobalScheduleEvent(time)]

    # ------------------------------------------------------------------
    # recovery scheduling
    # ------------------------------------------------------------------

    def _recover_requests(
        self, time: float, record: FailureRecord, affected: List[Request]
    ) -> list:
        from vidur.events.global_schedule_event import GlobalScheduleEvent

        to_route: List[Request] = []
        interrupted: List[InterruptedRequest] = []
        requests: Dict[int, Request] = {}

        for request in affected:
            num_processed_tokens, num_checkpointed_tokens, holder_replica_id = (
                record.snapshots.get(
                    request.id, (request.num_processed_tokens, 0, None)
                )
            )
            if num_processed_tokens == 0:
                # nothing lost, route it again like a new request
                record.num_requeued += 1
                to_route.append(request)
                continue

            info = InterruptedRequest(
                request_id=request.id,
                failed_replica_id=record.replica_id,
                failed_at=record.failed_at,
                num_processed_tokens=num_processed_tokens,
                num_checkpointed_tokens=num_checkpointed_tokens,
                holder_replica_id=holder_replica_id,
            )
            record.interrupted.append(info)
            self._awaiting_resume[request.id].append(info)
            self._interrupted_request_ids.add(request.id)
            interrupted.append(info)
            requests[request.id] = request

        if self._policy == FailureRecoveryPolicyType.STOP_AND_RESTART:
            plan = [(info, None, 0) for info in interrupted]
        elif self._policy == FailureRecoveryPolicyType.FIXED_CHECKPOINTING:
            plan = self._plan_fixed_holder(record.replica_id, interrupted)
        elif self._locality_aware_scheduling:
            plan = self._plan_locality_aware(interrupted)
        else:
            plan = self._plan_least_loaded(interrupted)

        events = []
        for info, target_replica_id, num_restored_tokens in plan:
            request = requests[info.request_id]
            info.target_replica_id = target_replica_id
            info.num_restored_tokens = num_restored_tokens
            request.restore_from_checkpoint(num_restored_tokens)
            if target_replica_id is None:
                to_route.append(request)
            else:
                events.append(
                    self._dispatch(
                        time, request, target_replica_id, num_restored_tokens
                    )
                )

        for request in to_route:
            self._scheduler.add_request(request)
        if to_route:
            events.append(GlobalScheduleEvent(time))

        return events

    def _plan_fixed_holder(
        self, failed_replica_id: int, interrupted: List[InterruptedRequest]
    ) -> list:
        # every interrupted request goes to the failed replica's fixed neighbour
        holder_replica_id = self._get_fixed_holder(failed_replica_id)
        if holder_replica_id is None or not self._is_routable(holder_replica_id):
            return [(info, None, 0) for info in interrupted]

        return [
            (info, holder_replica_id, info.num_checkpointed_tokens)
            for info in interrupted
        ]

    def _plan_least_loaded(self, interrupted: List[InterruptedRequest]) -> list:
        # recompute every interrupted request on the least loaded replica
        projected_load = self._get_projected_loads()
        plan = []
        for info in interrupted:
            target = self._get_least_loaded(projected_load)
            if target is not None:
                projected_load[target] += info.num_processed_tokens
            plan.append((info, target, 0))
        return plan

    def _plan_locality_aware(self, interrupted: List[InterruptedRequest]) -> list:
        """Resume each request on its checkpoint holder, redirecting the requests
        with the smallest checkpointed prefixes away from overloaded holders."""
        projected_load = self._get_projected_loads()
        if not projected_load:
            return [(info, None, 0) for info in interrupted]

        by_holder: Dict[int, List[InterruptedRequest]] = defaultdict(list)
        without_checkpoint: List[InterruptedRequest] = []
        for info in interrupted:
            if (
                info.num_checkpointed_tokens > 0
                and info.holder_replica_id in projected_load
            ):
                by_holder[info.holder_replica_id].append(info)
                projected_load[info.holder_replica_id] += info.num_processed_tokens
            else:
                without_checkpoint.append(info)

        plan = []
        for holder_replica_id, infos in by_holder.items():
            infos.sort(key=lambda info: info.num_checkpointed_tokens)
            for info in infos:
                if self._is_overloaded(
                    holder_replica_id, projected_load[holder_replica_id]
                ):
                    target = self._get_least_loaded(
                        projected_load, exclude=holder_replica_id
                    )
                    if target is not None and not self._is_overloaded(
                        target, projected_load[target] + info.num_processed_tokens
                    ):
                        projected_load[holder_replica_id] -= info.num_processed_tokens
                        projected_load[target] += info.num_processed_tokens
                        info.redirected = True
                        plan.append((info, target, 0))
                        continue

                plan.append((info, holder_replica_id, info.num_checkpointed_tokens))

        for info in without_checkpoint:
            target = self._get_least_loaded(projected_load)
            projected_load[target] += info.num_processed_tokens
            plan.append((info, target, 0))

        return plan

    def _dispatch(
        self,
        time: float,
        request: Request,
        target_replica_id: int,
        num_restored_tokens: int,
    ):
        from vidur.events.request_restore_end_event import RequestRestoreEndEvent

        self._assign(request, target_replica_id)

        restore_end = time
        if num_restored_tokens > 0:
            start = max(time, self._restore_free_at[target_replica_id])
            restore_end = start + num_restored_tokens * self._restore_seconds_per_token
            self._restore_free_at[target_replica_id] = restore_end

        # count the request's KV against the target until it is queued there
        num_reserved_tokens = request.num_prefill_tokens
        self._num_restoring_tokens[target_replica_id] += num_reserved_tokens

        return RequestRestoreEndEvent(
            restore_end,
            target_replica_id,
            request,
            self._scheduler.get_replica_scheduler(target_replica_id).epoch,
            num_reserved_tokens,
        )

    # ------------------------------------------------------------------
    # load estimates
    # ------------------------------------------------------------------

    def _get_load(self, replica_id: int) -> int:
        # KV tokens the replica holds or has been promised
        replica_scheduler = self._scheduler.get_replica_scheduler(replica_id)
        return (
            replica_scheduler.num_allocated_tokens
            + replica_scheduler.num_queued_tokens
            + self._num_restoring_tokens[replica_id]
        )

    def _get_projected_loads(self) -> Dict[int, int]:
        return {
            replica_id: self._get_load(replica_id)
            for replica_id in self._scheduler.routable_replica_ids
        }

    def _is_overloaded(self, replica_id: int, load: int) -> bool:
        capacity = self._scheduler.get_replica_scheduler(replica_id).kv_token_capacity
        return load > self._config.lumen_overload_threshold * capacity

    @staticmethod
    def _get_least_loaded(
        projected_load: Dict[int, int], exclude: Optional[int] = None
    ) -> Optional[int]:
        candidates = [
            (load, replica_id)
            for replica_id, load in projected_load.items()
            if replica_id != exclude
        ]
        return min(candidates)[1] if candidates else None

    # ------------------------------------------------------------------
    # KV checkpointing
    # ------------------------------------------------------------------

    def _get_fixed_holder(self, replica_id: int) -> Optional[int]:
        if self._num_replicas == 1:
            return None
        return (replica_id + 1) % self._num_replicas

    def _choose_holder(self, replica_id: int) -> Optional[int]:
        if not self._load_aware_checkpointing:
            return self._get_fixed_holder(replica_id)

        # the holder whose load would be lowest if this replica failed now
        candidates = [
            (
                self._get_load(holder_replica_id)
                + self._checkpoints.get_num_held_tokens(replica_id, holder_replica_id),
                holder_replica_id,
            )
            for holder_replica_id in self._scheduler.routable_replica_ids
            if holder_replica_id != replica_id
        ]
        return min(candidates)[1] if candidates else None

    def _stream_checkpoint(
        self, time: float, replica_id: int, request: Request
    ) -> None:
        checkpoint = self._checkpoints.get(request.id)
        if checkpoint is None or (
            self._load_aware_checkpointing
            and not self._is_alive(checkpoint.holder_replica_id)
        ):
            holder_replica_id = self._choose_holder(replica_id)
            if holder_replica_id is None:
                return
            checkpoint = self._checkpoints.assign(
                request.id, replica_id, holder_replica_id
            )

        if not self._is_alive(checkpoint.holder_replica_id):
            # a fixed holder that is down: resume streaming once it is back
            return

        self._checkpoints.stream(time, checkpoint, request.num_processed_tokens)

    # ------------------------------------------------------------------
    # speculation-assisted progressive recovery
    # ------------------------------------------------------------------

    def _continue_speculation(self, time: float, replica_id: int, batch: Batch) -> list:
        from vidur.events.speculative_draft_end_event import SpeculativeDraftEndEvent

        if not self._speculative_requests and self._num_speculative_slots == 0:
            return []

        new_requests = []
        continuing_requests = []
        for request in batch.requests:
            if request.completed or not request.is_prefill_complete:
                self._stop_speculating(request)
            elif request.id in self._speculative_requests:
                if len(self._speculative_requests) > self._num_speculative_slots:
                    # not enough draft capacity left for everyone
                    self._stop_speculating(request)
                else:
                    continuing_requests.append(request)
            elif len(self._speculative_requests) < self._num_speculative_slots:
                self._speculative_requests[request.id] = request
                new_requests.append(request)

        if not new_requests and not continuing_requests:
            return []

        replica_scheduler = self._scheduler.get_replica_scheduler(replica_id)
        replica_scheduler.hold_requests(new_requests + continuing_requests)
        epoch = replica_scheduler.epoch

        draft_time = (
            self._config.speculative_num_draft_tokens
            * self._config.draft_token_latency_s
        )
        events = []
        if continuing_requests:
            events.append(
                SpeculativeDraftEndEvent(
                    time + draft_time, replica_id, continuing_requests, epoch
                )
            )
        for request in new_requests:
            # the draft model first prefills the request's context
            context_time = (
                request.num_processed_tokens
                * self._config.draft_prefill_time_per_token_s
            )
            events.append(
                SpeculativeDraftEndEvent(
                    time + context_time + draft_time, replica_id, [request], epoch
                )
            )
        return events

    def _stop_speculating(self, request: Request) -> None:
        self._speculative_requests.pop(request.id, None)
        request.num_speculative_tokens = 0

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _is_alive(self, replica_id: Optional[int]) -> bool:
        return (
            replica_id is not None
            and self._scheduler.get_replica_scheduler(replica_id).is_alive
        )

    def _is_routable(self, replica_id: int) -> bool:
        return replica_id in self._scheduler.routable_replica_ids

    def _assign(self, request: Request, replica_id: int) -> None:
        self._unassign(request)
        self._request_replica[request.id] = replica_id
        self._replica_requests[replica_id][request.id] = request

        checkpoint = self._checkpoints.get(request.id)
        if checkpoint and checkpoint.source_replica_id != replica_id:
            # KV checkpoints follow the replica that runs the request
            self._checkpoints.drop(request.id)

    def _unassign(self, request: Request) -> None:
        replica_id = self._request_replica.pop(request.id, None)
        if replica_id is not None:
            self._replica_requests[replica_id].pop(request.id, None)

    def _check_resumed(self, time: float, request: Request) -> None:
        infos = self._awaiting_resume.get(request.id)
        if not infos:
            return

        for info in infos:
            if request.num_processed_tokens > info.num_processed_tokens:
                info.resumed_at = time

        infos[:] = [info for info in infos if info.resumed_at is None]
        if not infos:
            del self._awaiting_resume[request.id]

    def _record_output_tokens(self, time: float, request: Request) -> None:
        # tokens recomputed after a failure were already delivered once
        num_output_tokens = (
            request.num_processed_tokens - request.initial_num_prefill_tokens
        )
        watermark = self._output_token_watermark.get(request.id, 0)
        if num_output_tokens > watermark:
            self._num_output_tokens_per_second[int(time)] += (
                num_output_tokens - watermark
            )
            self._output_token_watermark[request.id] = num_output_tokens

    def _on_request_completed(self, request: Request) -> None:
        self._unassign(request)
        self._checkpoints.drop(request.id)
        self._output_token_watermark.pop(request.id, None)

        first_token_at = request.prefill_completed_at
        num_decode_tokens = request.initial_num_decode_tokens
        self._completed_requests.append(
            {
                "request_id": request.id,
                "arrived_at": request.arrived_at,
                "first_token_at": first_token_at,
                "completed_at": request.completed_at,
                "num_prefill_tokens": request.initial_num_prefill_tokens,
                "num_decode_tokens": num_decode_tokens,
                "ttft": first_token_at - request.arrived_at,
                "tpot": (
                    (request.completed_at - first_token_at) / (num_decode_tokens - 1)
                    if num_decode_tokens > 1
                    else 0.0
                ),
                "e2e_time": request.completed_at - request.arrived_at,
                "num_restarts": request.num_restarts,
                "interrupted": request.id in self._interrupted_request_ids,
            }
        )

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------

    def write_report(self) -> Optional[dict]:
        if not self._enabled:
            return None

        os.makedirs(self._output_dir, exist_ok=True)

        requests_df = pd.DataFrame(self._completed_requests)
        if not requests_df.empty:
            requests_df.sort_values("request_id").to_csv(
                f"{self._output_dir}/failover_request_metrics.csv", index=False
            )

        timeline = sorted(self._num_output_tokens_per_second.items())
        pd.DataFrame(timeline, columns=["time_s", "output_tokens"]).to_csv(
            f"{self._output_dir}/failover_output_tokens_per_second.csv", index=False
        )

        report = {
            "recovery_policy": str(self._policy),
            "lumen_mechanisms": {
                "load_aware_checkpointing": self._load_aware_checkpointing,
                "locality_aware_scheduling": self._locality_aware_scheduling,
                "speculative_recovery": self._speculative_recovery,
            },
            "kv_cache_bytes_per_token": self._kv_bytes_per_token,
            "model_reload_time": self._model_reload_time,
            "num_checkpoint_tokens_streamed": self._checkpoints.num_streamed_tokens,
            "num_checkpoint_bytes_streamed": self._checkpoints.num_streamed_tokens
            * self._kv_bytes_per_token,
            "speculation": {
                "num_verification_steps": self._num_verification_steps,
                "num_draft_tokens": self._num_draft_tokens,
                "num_accepted_draft_tokens": self._num_accepted_draft_tokens,
                "mean_tokens_per_verification": (
                    1 + self._num_accepted_draft_tokens / self._num_verification_steps
                    if self._num_verification_steps
                    else None
                ),
            },
            "failures": [record.to_dict() for record in self._failures],
            "requests": self._summarize_requests(requests_df),
        }

        with open(f"{self._output_dir}/failover_report.json", "w") as f:
            json.dump(report, f, indent=4)

        self._log_report(report)
        return report

    def _summarize_requests(self, requests_df: pd.DataFrame) -> dict:
        if requests_df.empty:
            return {}

        # interrupted requests and those arriving while a replica is down
        affected = requests_df["interrupted"].copy()
        for record in self._failures:
            end = record.recovered_at if record.recovered_at is not None else np.inf
            affected |= requests_df["arrived_at"].between(record.failed_at, end)

        groups = {
            "all": requests_df,
            "interrupted": requests_df[requests_df["interrupted"]],
            "affected": requests_df[affected],
        }
        return {name: summarize_latencies(df) for name, df in groups.items()}

    def _log_report(self, report: dict) -> None:
        for failure in report["failures"]:
            logger.info(
                f"Failure of replica {failure['replica_id']} at {failure['failed_at']:.1f}s: "
                f"{failure['num_interrupted_requests']} interrupted, "
                f"{failure['num_restored_from_checkpoint']} restored from checkpoint, "
                f"{failure['num_redirected']} redirected, "
                f"recovery time {failure['recovery_time']}"
            )
        overall = report["requests"].get("all")
        if overall:
            logger.info(
                f"Mean TTFT {overall['ttft']['mean']:.3f}s, "
                f"mean TPOT {overall['tpot']['mean'] * 1e3:.1f}ms "
                f"over {overall['num_requests']} requests"
            )
