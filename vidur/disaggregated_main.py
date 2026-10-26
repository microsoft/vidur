import json

from vidur.config import DisaggregatedSimulationConfig
from vidur.entities import Replica
from vidur.execution_time_predictor import ExecutionTimePredictorRegistry
from vidur.logger import init_logger
from vidur.request_generator import RequestGeneratorRegistry
from vidur.scheduler.disaggregated_scheduler import (
    DisaggregatedScheduler,
    build_time_fns_from_predictor,
    kv_cache_bytes_per_token,
)
from vidur.scheduler.utils.memory_planner import MemoryPlanner
from vidur.utils.random import set_seeds

logger = init_logger(__name__)


def main() -> None:
    config: DisaggregatedSimulationConfig = (
        DisaggregatedSimulationConfig.create_from_cli_args()
    )
    set_seeds(config.seed)

    replica_config = config.cluster_config.replica_config
    replica_scheduler_config = config.cluster_config.replica_scheduler_config
    assert (
        replica_config.num_pipeline_stages == 1
    ), "Disaggregated simulation supports pipeline_parallel_size=1 only."

    execution_time_predictor = ExecutionTimePredictorRegistry.get(
        config.execution_time_predictor_config.get_type(),
        predictor_config=config.execution_time_predictor_config,
        replica_config=replica_config,
        replica_scheduler_config=replica_scheduler_config,
        metrics_config=config.metrics_config,
    )
    prefill_time_fn, decode_step_time_fn = build_time_fns_from_predictor(
        execution_time_predictor, config.execution_time_predictor_config
    )

    # same decode batch limit as BaseReplicaScheduler
    replica = Replica(replica_config, config.request_generator_config)
    max_decode_batch_size = min(
        MemoryPlanner(replica_config, replica).get_max_batch_size(),
        replica_scheduler_config.batch_size_cap,
    )

    requests = RequestGeneratorRegistry.get(
        config.request_generator_config.get_type(),
        config.request_generator_config,
    ).generate()

    scheduler = DisaggregatedScheduler(
        num_prefill_replicas=config.num_prefill_replicas,
        num_decode_replicas=config.num_decode_replicas,
        prefill_time_fn=prefill_time_fn,
        decode_step_time_fn=decode_step_time_fn,
        kv_bytes_per_token=kv_cache_bytes_per_token(replica_config.model_config),
        kv_transfer_bandwidth_gb_per_s=config.kv_transfer_bandwidth_gb_per_s,
        max_decode_batch_size=max_decode_batch_size,
    )
    result = scheduler.simulate(
        [(r.arrived_at, r.num_prefill_tokens, r.num_decode_tokens) for r in requests]
    )

    output_file = f"{config.metrics_config.output_dir}/disaggregated_metrics.json"
    with open(output_file, "w") as f:
        json.dump(result.to_dict(), f, indent=4)
    logger.info(f"Wrote disaggregated simulation metrics to {output_file}")


if __name__ == "__main__":
    main()
