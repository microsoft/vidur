# Failure Recovery Simulation (LUMEN)

Vidur can inject worker failures and simulate the recovery strategies compared in
[LUMEN: Coordinated Failure Recovery for Distributed LLM Serving](https://arxiv.org/abs/2606.17787)
(Cao et al., 2026):

| Policy | `--failure_config_recovery_policy` | What happens to interrupted requests |
| --- | --- | --- |
| Stop-and-Restart | `stop_and_restart` | Re-dispatched through the global scheduler to surviving replicas and recomputed from scratch. |
| Fixed-Checkpointing | `fixed_checkpointing` | Every request streams its KV cache to the host memory of a fixed neighbour, replica `(r + 1) mod N` (as in DéjàVu). After a failure, all interrupted requests resume on that neighbour from their checkpoints. |
| LUMEN | `lumen` | Load-aware KV checkpointing, locality-aware recovery scheduling and speculation-assisted progressive recovery (below). |

The failure model and LUMEN's mechanisms are implemented from the paper's
description. Where the paper's exact formulas were not available, the choices
made here are spelled out below and exposed as parameters.

## Failure model

A LUMEN *worker* is a Vidur *replica*. With pipeline parallelism, a replica is a
whole pipeline: a failure of any stage takes the entire replica down, because
the serving engine restarts as a unit.

1. **Failure** (`failure_times`, `failure_replica_ids`, `failure_stage_ids`):
   batches in flight on the replica are discarded, its GPU-resident KV cache is
   lost, and checkpoint transfers that had not finished are lost.
   KV checkpoints the replica held for other replicas are lost too, unless
   `--no-failure_config_holder_failure_loses_checkpoints` is set (GPU-only
   failure with host memory intact).
2. **Detection**, `detection_delay_s` later: until then the global scheduler
   keeps routing new requests to the failed replica. At detection the replica
   stops receiving requests and recovery starts:
   * requests without progress (queued, or routed during the detection gap)
     are routed again like new requests,
   * requests that had made progress are *interrupted* and handled by the
     recovery policy.
3. **Reload**: the replica comes back `model_reload_time_s` after detection.
   If unset, this is `engine_restart_overhead_s` plus the per-device weight
   bytes divided by `weight_load_bandwidth_gb_per_s`, as all stages load their
   shards in parallel. For Qwen-72B with PP4/TP1 this is 15 s + 34.9 GB / 4 GB/s = 23.7 s.

## KV checkpointing (fixed_checkpointing and lumen)

After each iteration, a request's new KV tokens are streamed to its holder in
units of `checkpoint_granularity_tokens`. Each replica has one outgoing link of
`checkpoint_bandwidth_gb_per_s`, shared first-come-first-served by all its
requests, so checkpoints lag behind the GPU cache when the link is busy. Only
tokens whose transfer finished before the failure survive it.

When an interrupted request resumes on its holder, the checkpoint is copied from
host memory to the holder's GPUs at `restore_bandwidth_gb_per_s_per_device`
(every device copies its own layers in parallel, and copies to one replica are
serialized). Tokens processed after the last durable checkpoint are recomputed
as a prefill on top of the restored prefix. A request whose replica changes
starts a new checkpoint from scratch.

Checkpointing is assumed to overlap with compute, so it does not slow down
iterations, and holder host memory is assumed to be large enough.

## LUMEN

All three mechanisms use the same load estimate: the KV tokens a replica holds
or has been promised, i.e. allocated KV blocks, plus the prompts in its queue,
plus checkpoints currently being restored into it.

**Load-aware KV checkpointing** (`lumen_load_aware_checkpointing`). When a request
on replica `s` first needs a checkpoint, it is placed on the alive replica `h != s` that minimizes

    load(h) + checkpointed_tokens(s -> h)

which is the load `h` would carry if `s` failed now. A request whose holder
fails is placed again. Disabled, checkpoints go to the fixed neighbour.

**Locality-aware recovery scheduling** (`lumen_locality_aware_scheduling`).
Interrupted requests with a usable checkpoint are sent to their holder. While a
holder's projected load exceeds `lumen_overload_threshold` of its KV capacity,
the requests with the *smallest* checkpointed prefixes are redirected, without
their checkpoint, to the least-loaded other replica, as long as that replica
stays under the threshold. Requests without a checkpoint go to the least-loaded
replica. Disabled, all interrupted requests are recomputed on the least-loaded
replicas (no KV reuse).

**Speculation-assisted progressive recovery** (`lumen_speculative_recovery`).
`draft_model_load_time_s` after detection, the recovering replica hosts a draft
model and serves as the drafter for up to `speculative_max_requests` decode
requests on the surviving replicas, until its full model is reloaded:

* before each verification the request waits for its drafts:
  `speculative_num_draft_tokens * draft_token_latency_s`, plus
  `draft_prefill_time_per_token_s` per context token the first time,
* the surviving replica verifies `k + 1` tokens in one iteration (costed like a
  decode with `k + 1` tokens for the linear layers and a decode for attention),
* each draft token is accepted with probability `speculative_acceptance_rate`
  until the first rejection, and the request advances by the accepted tokens
  plus one.

## Outputs

Besides Vidur's usual metrics, each run with a recovery policy writes to its output directory:

* `failover_report.json`: per failure, the detection/draft/reload times, number
  of interrupted and requeued requests, how many resumed from a checkpoint or
  were redirected, checkpointed vs. recomputed tokens, and the *recovery time*,
  i.e. the time until every interrupted request has progressed past where it was
  when the replica failed. It also has speculation statistics and TTFT, TPOT and
  end-to-end latency (mean, p50, p90, p99) over all requests, the interrupted
  requests, and the *affected* requests (interrupted, or arriving between a
  failure and the end of the reload).
* `failover_request_metrics.csv`: per-request TTFT, TPOT and end-to-end time.
  TPOT is `(completion - first token) / (output tokens - 1)` using the request's
  original length, so it is not distorted by restarts.
* `failover_output_tokens_per_second.csv`: output tokens delivered per second.
  Recomputed tokens are not counted twice.

`python -m vidur.failover.compare label=dir ...` prints a comparison table and,
with `--output_dir`, writes it along with a throughput-over-time plot.
`--group interrupted|affected` restricts the latency columns to those requests.
The set is taken from the runs with failures and matched by request id, so a run
without failures (same seed and workload) serves as the reference.

## Example: two Qwen-72B pipelines

[`examples/failover/qwen72b_2x_pp4.sh`](../examples/failover/qwen72b_2x_pp4.sh)
serves Qwen-72B on two replicas, each a 4-stage pipeline (TP1) on a 4xA100
node, with Poisson arrivals and Splitwise conversation lengths. Stage 2 of
replica 0 fails at t = 300 s. It runs every policy plus two LUMEN ablations and
compares them:

```sh
bash examples/failover/qwen72b_2x_pp4.sh
QPS=1.2 FAILURE_TIME=200 OUT=simulator_output/qps1.2 bash examples/failover/qwen72b_2x_pp4.sh
```

A single run looks like this:

```sh
python -m vidur.main \
  --replica_config_model_name Qwen/Qwen-72B \
  --cluster_config_num_replicas 2 \
  --replica_config_num_pipeline_stages 4 \
  --global_scheduler_config_type round_robin \
  --failure_config_recovery_policy lumen \
  --failure_config_failure_times 300 \
  --failure_config_failure_replica_ids 0 \
  --failure_config_failure_stage_ids 2
```

With only two replicas, every request's checkpoint holder is necessarily the
other replica and there is nowhere to redirect to. LUMEN's load-aware placement
and locality-aware scheduling therefore behave like Fixed-Checkpointing, and the
difference between the two comes from speculative recovery. Use more replicas
(`--cluster_config_num_replicas`) to see placement and redirection matter.

Vidur's `lor` global scheduler only counts queued requests and breaks ties
towards replica 0, so it can leave replicas very unevenly loaded at moderate
load; the example uses `round_robin`. Failed replicas are skipped by all global
schedulers.

## Parameters

All parameters are `--failure_config_<name>`:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `recovery_policy` | `none` | `none`, `stop_and_restart`, `fixed_checkpointing` or `lumen`. |
| `failure_times` | | Failure times in seconds, e.g. `300 600`. |
| `failure_replica_ids` | all `0` | Replica failing at each time. |
| `failure_stage_ids` | all `0` | Pipeline stage failing at each time (reported only). |
| `detection_delay_s` | 1.0 | Failure to detection. |
| `model_reload_time_s` | derived | Detection to replica back in service. |
| `engine_restart_overhead_s` | 15.0 | Fixed part of the reload time. |
| `weight_load_bandwidth_gb_per_s` | 4.0 | Per-device weight loading bandwidth. |
| `checkpoint_bandwidth_gb_per_s` | 25.0 | Per-replica KV checkpoint streaming bandwidth. |
| `checkpoint_granularity_tokens` | 16 | Tokens per checkpoint transfer. |
| `restore_bandwidth_gb_per_s_per_device` | 20.0 | Host-to-GPU copy bandwidth when restoring. |
| `holder_failure_loses_checkpoints` | true | A failed replica loses the checkpoints it holds. |
| `lumen_load_aware_checkpointing` | true | LUMEN mechanism (i). |
| `lumen_locality_aware_scheduling` | true | LUMEN mechanism (ii). |
| `lumen_speculative_recovery` | true | LUMEN mechanism (iii). |
| `lumen_overload_threshold` | 0.9 | Fraction of KV capacity above which a holder is overloaded. |
| `draft_model_load_time_s` | 3.0 | Detection to draft model ready. |
| `speculative_num_draft_tokens` | 4 | Draft tokens per verification. |
| `speculative_acceptance_rate` | 0.7 | Per-token acceptance probability. |
| `draft_token_latency_s` | 0.008 | Draft model time per drafted token. |
| `draft_prefill_time_per_token_s` | 0.00002 | Draft model time per context token. |
| `speculative_max_requests` | 128 | Requests one recovering replica can draft for. |

Failure recovery supports the `sarathi` and `vllm` replica schedulers.
