from simulator.scheduler.replica_scheduler.dsarathi_replica_scheduler import (
    DSarathiReplicaScheduler,
)
from simulator.scheduler.replica_scheduler.faster_transformer_replica_scheduler import (
    FasterTransformerReplicaScheduler,
)
from simulator.scheduler.replica_scheduler.orca_replica_scheduler import (
    OrcaReplicaScheduler,
)
from simulator.scheduler.replica_scheduler.sarathi_replica_scheduler import (
    SarathiReplicaScheduler,
)
from simulator.scheduler.replica_scheduler.vllm_replica_scheduler import (
    VLLMReplicaScheduler,
)
from simulator.types import ReplicaSchedulerType
from simulator.utils.base_registry import BaseRegistry


class ReplicaSchedulerRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> ReplicaSchedulerType:
        return ReplicaSchedulerType.from_str(key_str)


ReplicaSchedulerRegistry.register(
    ReplicaSchedulerType.FASTER_TRANSFORMER, FasterTransformerReplicaScheduler
)
ReplicaSchedulerRegistry.register(ReplicaSchedulerType.ORCA, OrcaReplicaScheduler)
ReplicaSchedulerRegistry.register(ReplicaSchedulerType.SARATHI, SarathiReplicaScheduler)
ReplicaSchedulerRegistry.register(ReplicaSchedulerType.VLLM, VLLMReplicaScheduler)
ReplicaSchedulerRegistry.register(
    ReplicaSchedulerType.DSARATHI, DSarathiReplicaScheduler
)
