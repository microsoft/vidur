#!/usr/bin/env bash
# Failover study from LUMEN (arXiv:2606.17787) on two Qwen-72B pipelines.
#
# Each replica is a 4-stage pipeline (PP4, TP1) on a 4xA100 node. One stage of
# replica 0 fails at FAILURE_TIME; the run is repeated for every recovery
# policy and the results are compared. See docs/failover.md.
#
# Usage (from the repository root):
#   bash examples/failover/qwen72b_2x_pp4.sh
#   QPS=1.2 FAILURE_TIME=200 bash examples/failover/qwen72b_2x_pp4.sh
set -euo pipefail

OUT=${OUT:-simulator_output/failover_qwen72b_2x_pp4}
QPS=${QPS:-1.0}
NUM_REQUESTS=${NUM_REQUESTS:-1200}
FAILURE_TIME=${FAILURE_TIME:-300}
FAILED_REPLICA=${FAILED_REPLICA:-0}
FAILED_STAGE=${FAILED_STAGE:-2}
SEED=${SEED:-42}

export WANDB_MODE=${WANDB_MODE:-disabled}

COMMON_ARGS=(
  --seed "$SEED"
  --replica_config_model_name Qwen/Qwen-72B
  --replica_config_device a100
  --replica_config_network_device a100_pairwise_nvlink
  --cluster_config_num_replicas 2
  --replica_config_num_pipeline_stages 4
  --replica_config_tensor_parallel_size 1
  # Vidur's LOR router only counts queued requests and favours replica 0,
  # round robin keeps both pipelines equally loaded before the failure
  --global_scheduler_config_type round_robin
  --replica_scheduler_config_type sarathi
  --sarathi_scheduler_config_chunk_size 512
  --sarathi_scheduler_config_batch_size_cap 128
  --request_generator_config_type synthetic
  --synthetic_request_generator_config_num_requests "$NUM_REQUESTS"
  --length_generator_config_type trace
  --trace_request_length_generator_config_trace_file ./data/processed_traces/splitwise_conv.csv
  --trace_request_length_generator_config_max_tokens 4096
  --interval_generator_config_type poisson
  --poisson_request_interval_generator_config_qps "$QPS"
  --no-metrics_config_store_plots
  --no-metrics_config_enable_chrome_trace
)

FAILURE_ARGS=(
  --failure_config_failure_times "$FAILURE_TIME"
  --failure_config_failure_replica_ids "$FAILED_REPLICA"
  --failure_config_failure_stage_ids "$FAILED_STAGE"
)

run() {
  local name=$1
  shift
  echo "=== $name"
  python -m vidur.main "${COMMON_ARGS[@]}" \
    --metrics_config_output_dir "$OUT/$name" "$@"
}

# stop_and_restart without failures is plain serving, used as the reference
run no_failure --failure_config_recovery_policy stop_and_restart
run stop_and_restart --failure_config_recovery_policy stop_and_restart "${FAILURE_ARGS[@]}"
run fixed_checkpointing --failure_config_recovery_policy fixed_checkpointing "${FAILURE_ARGS[@]}"
run lumen --failure_config_recovery_policy lumen "${FAILURE_ARGS[@]}"

# LUMEN ablations: no speculation, and speculation without KV reuse
run lumen_no_speculation --failure_config_recovery_policy lumen "${FAILURE_ARGS[@]}" \
  --no-failure_config_lumen_speculative_recovery
run lumen_speculation_only --failure_config_recovery_policy lumen "${FAILURE_ARGS[@]}" \
  --no-failure_config_lumen_load_aware_checkpointing \
  --no-failure_config_lumen_locality_aware_scheduling

for group in all affected; do
  echo "=== latency over $group requests"
  python -m vidur.failover.compare --group "$group" \
    --output_dir "$OUT/comparison_$group" \
    no_failure="$OUT/no_failure" \
    stop_and_restart="$OUT/stop_and_restart" \
    fixed_checkpointing="$OUT/fixed_checkpointing" \
    lumen="$OUT/lumen" \
    lumen_no_speculation="$OUT/lumen_no_speculation" \
    lumen_speculation_only="$OUT/lumen_speculation_only"
done
