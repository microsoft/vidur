from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class InterruptedRequest:
    """A request that had made progress on a replica when the replica failed."""

    request_id: int
    failed_replica_id: int
    failed_at: float
    num_processed_tokens: int
    num_checkpointed_tokens: int
    holder_replica_id: Optional[int]
    target_replica_id: Optional[int] = None
    # tokens reused from the checkpoint; the rest of the processed tokens are recomputed
    num_restored_tokens: int = 0
    redirected: bool = False
    # first batch end after which the request is past its pre-failure progress
    resumed_at: Optional[float] = None

    @property
    def num_lost_tokens(self) -> int:
        return self.num_processed_tokens - self.num_restored_tokens


@dataclass
class FailureRecord:
    replica_id: int
    stage_id: int
    failed_at: float
    detected_at: Optional[float] = None
    draft_ready_at: Optional[float] = None
    recovered_at: Optional[float] = None
    # request id -> (processed tokens, durable checkpointed tokens, holder) at failure
    snapshots: Dict[int, Tuple[int, int, Optional[int]]] = field(default_factory=dict)
    interrupted: List[InterruptedRequest] = field(default_factory=list)
    # requests without progress on the failed replica, routed again as new
    num_requeued: int = 0

    def to_dict(self) -> dict:
        resume_latencies = [
            r.resumed_at - self.failed_at
            for r in self.interrupted
            if r.resumed_at is not None
        ]
        return {
            "replica_id": self.replica_id,
            "stage_id": self.stage_id,
            "failed_at": self.failed_at,
            "detected_at": self.detected_at,
            "draft_ready_at": self.draft_ready_at,
            "recovered_at": self.recovered_at,
            "capacity_restoration_time": (
                self.recovered_at - self.failed_at
                if self.recovered_at is not None
                else None
            ),
            "num_interrupted_requests": len(self.interrupted),
            "num_requeued_requests": self.num_requeued,
            "num_restored_from_checkpoint": sum(
                1 for r in self.interrupted if r.num_restored_tokens > 0
            ),
            "num_redirected": sum(1 for r in self.interrupted if r.redirected),
            "num_processed_tokens": sum(
                r.num_processed_tokens for r in self.interrupted
            ),
            "num_checkpointed_tokens": sum(
                r.num_checkpointed_tokens for r in self.interrupted
            ),
            "num_restored_tokens": sum(r.num_restored_tokens for r in self.interrupted),
            "num_recomputed_tokens": sum(r.num_lost_tokens for r in self.interrupted),
            "num_unresumed_requests": len(self.interrupted) - len(resume_latencies),
            # time until every interrupted request is past its pre-failure progress
            "recovery_time": max(resume_latencies) if resume_latencies else None,
            "mean_request_recovery_latency": (
                sum(resume_latencies) / len(resume_latencies)
                if resume_latencies
                else None
            ),
            "requests_per_target_replica": self._count_targets(),
        }

    def _count_targets(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in self.interrupted:
            key = (
                "global_scheduler"
                if r.target_replica_id is None
                else str(r.target_replica_id)
            )
            counts[key] = counts.get(key, 0) + 1
        return counts
