# Disaggregated Prefill/Decode Scheduling

VIDUR's `DisaggregatedScheduler` simulates separate prefill and decode worker
fleets (Dynamo-style P/D disaggregation), enabling sweep-based optimization of
the prefill-to-decode worker ratio for a given model, traffic pattern, and
interconnect.

## Motivation

In collocated serving, every replica handles both prefill and decode.
Disaggregated serving dedicates some replicas exclusively to prefill and others
to decode:

```
request → prefill_replica → [KV transfer] → decode_replica → response
```

The optimal split depends on the prompt/output length distribution, GPU
compute characteristics, and interconnect bandwidth. VIDUR can sweep this
ratio without running real hardware.

## How it works

The scheduler is a discrete-event simulation with four queues:

```
arrival_queue → prefill_queue → kv_transfer_queue → decode_queue → done
```

**KV transfer latency** — the key new latency term vs collocated serving:

```
t_kv = kv_bytes / interconnect_bandwidth

kv_bytes = 2 × num_layers × num_kv_heads × head_dim × seq_len × 2 bytes (fp16)
```

For DeepSeek-V3 (61 layers, 128 KV heads, head_dim=56, seq_len=1024):
- NVLink (600 GB/s): ~0.13 ms
- InfiniBand HDR (400 GB/s): ~0.20 ms
- PCIe 4.0 ×16 (64 GB/s): ~1.25 ms

## Usage

```python
from vidur.config import ReplicaConfig
from vidur.scheduler.disaggregated_scheduler import (
    DisaggregatedScheduler,
    DisaggregatedReplicaSchedulerConfig,
)

sched_config = DisaggregatedReplicaSchedulerConfig(
    prefill_fleet_size=2,
    decode_fleet_size=4,
    interconnect_bandwidth_gbps=400.0,  # InfiniBand
)
replica_config = ReplicaConfig(
    model_name="deepseek-ai/DeepSeek-V3",
    device="a100",
)
scheduler = DisaggregatedScheduler(replica_config, sched_config)
result = scheduler.simulate(requests)

print(f"P99 E2E latency: {result.e2e_latency_p99_ms:.1f} ms")
print(f"TTFT P50:        {result.ttft_p50_ms:.1f} ms")
print(f"KV transfer avg: {result.kv_transfer_mean_ms:.1f} ms")
```

## P/D ratio sweep

```python
for p_frac in [0.2, 0.3, 0.4, 0.5]:
    fleet = 8
    p = max(1, int(fleet * p_frac))
    d = fleet - p
    cfg = DisaggregatedReplicaSchedulerConfig(
        prefill_fleet_size=p, decode_fleet_size=d,
        interconnect_bandwidth_gbps=600.0,
    )
    result = DisaggregatedScheduler(replica_config, cfg).simulate(requests)
    print(f"p={p} d={d}: p99={result.e2e_latency_p99_ms:.0f}ms  ttft_p50={result.ttft_p50_ms:.0f}ms")
```

## Configuration reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prefill_fleet_size` | 1 | Number of prefill-only replicas |
| `decode_fleet_size` | 1 | Number of decode-only replicas |
| `interconnect_bandwidth_gbps` | 400.0 | NVLink=600, InfiniBand=400, PCIe=64 |

## MoE models

DeepSeek-V3 and Mixtral use `BaseMoEModelConfig`, which adds:
- `num_experts` / `num_active_experts` (top-k routing)
- `expert_intermediate_dim`
- `kv_lora_rank` / `q_lora_rank` (MLA attention for DeepSeek-V3)

The `MoELayerExecutionTimePredictor` extends the standard RF predictor with
load-imbalance correction (λ^0.72) using a closed-form multinomial approximation
for expected maximum expert load.

To use DeepSeek-V3:

```bash
python -m vidur.main \
  --replica_config_model_name deepseek-ai/DeepSeek-V3 \
  --replica_config_device a100 \
  --cluster_config_num_replicas 1 \
  --replica_config_tensor_parallel_size 8
```
