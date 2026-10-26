# Disaggregated Prefill/Decode Simulation

`vidur.disaggregated_main` simulates disaggregated serving, where prefill and
decode run on separate replica fleets (Splitwise, DistServe, Dynamo). It answers
questions such as "how should 8 GPUs be split between prefill and decode for
this workload and interconnect?".

```
request -> prefill replica -> KV transfer -> decode replica -> response
```

Prefill and decode step times come from the same trained execution time
predictor as `vidur.main`, so any model/device pair with profiling data in
`data/profiling` can be simulated.

## Usage

All regular Vidur flags apply (model, device, tensor parallelism, request
generator, execution time predictor, scheduler batch size cap). Three flags
are specific to disaggregation:

| Flag | Meaning |
|------|---------|
| `--num_prefill_replicas` | prefill-only replicas |
| `--num_decode_replicas` | decode-only replicas |
| `--kv_transfer_bandwidth_gb_per_s` | outbound KV bandwidth per prefill replica, GB/s, aggregated over its TP workers |

```sh
python -m vidur.disaggregated_main \
--replica_config_device a100 \
--replica_config_model_name meta-llama/Meta-Llama-3-8B \
--replica_config_tensor_parallel_size 1 \
--request_generator_config_type synthetic \
--synthetic_request_generator_config_num_requests 256 \
--length_generator_config_type trace \
--trace_request_length_generator_config_max_tokens 4096 \
--trace_request_length_generator_config_trace_file ./data/processed_traces/splitwise_conv.csv \
--interval_generator_config_type poisson \
--poisson_request_interval_generator_config_qps 4 \
--replica_scheduler_config_type vllm \
--vllm_scheduler_config_batch_size_cap 128 \
--num_prefill_replicas 1 \
--num_decode_replicas 1 \
--kv_transfer_bandwidth_gb_per_s 50
```

Metrics (TTFT, TPOT, end-to-end latency and KV transfer time as mean/p50/p90/p99,
throughput, per-fleet utilization, decode batch sizes) are written to
`disaggregated_metrics.json` in the metrics output directory. To find the best
split, sweep `--num_prefill_replicas`/`--num_decode_replicas` at a fixed total.

## Model

* **Prefill**: each prefill replica runs one request at a time, FIFO. Prefill
  completion produces the first output token, so TTFT is measured there.
* **KV transfer**: each prefill replica has one outbound link of
  `kv_transfer_bandwidth_gb_per_s`. Transfers leaving the same replica are
  serialised, so link queueing shows up in the reported KV transfer time.
  KV bytes per token are `2 * num_kv_heads * head_dim * num_layers * 2` for
  MHA/GQA models and `(kv_lora_rank + qk_rope_head_dim) * num_layers * 2` for
  MLA models such as DeepSeek-V3 (70,272 bytes/token, about 25x less than the
  per-head formula would give).
* **Decode**: continuous batching. Every iteration emits one token for each
  running request; queued requests join at iteration boundaries. The running
  batch is capped by the same rule the replica schedulers use:
  `min(memory planner limit, batch_size_cap)`. The iteration time is the
  predictor's decode time for the batch size and average KV length.

## Limitations

* Pipeline parallelism is not supported (`num_pipeline_stages` must be 1).
* No chunked or batched prefill on prefill replicas, and no preemption.
* KV transfer is modelled as bandwidth only (no per-transfer setup latency and
  no contention on the decode side).
* `deepseek-ai/DeepSeek-V3` and `mistralai/Mixtral-8x7B-v0.1` configs are
  included for KV sizing, but Vidur has no MoE operator profiling data yet, so
  they cannot be simulated end to end.
