# Priority Distribution System

## Overview

The simulator now supports configurable priority distributions for requests, aligned with the Llumnix paper's priority-aware scheduling. Priorities range from 0 (critical/highest) to `num_priority_levels - 1` (background/lowest).

## Priority Semantics

For a 5-level configuration:
- **Priority 0 (Critical)**: Highest priority, largest execution headroom (2400 tokens default)
- **Priority 1 (High)**: High priority, moderate headroom (1600 tokens default)
- **Priority 2 (Normal)**: Normal priority, no additional headroom
- **Priority 3 (Low)**: Low priority, no headroom
- **Priority 4 (Background)**: Lowest priority, no headroom

## Configuration

Configure priorities in the synthetic request generator:

```bash
python3 -m vidur.main \
  --global_scheduler_config_type llumnix \
  --synthetic_request_generator_config_num_priority_levels 5 \
  --synthetic_request_generator_config_priority_distribution_type 3 \
  --metrics_config_enable_chrome_trace
```

## Available Distributions

### 1. ROUND_ROBIN (type=1)
- **Description**: Cycles through priority levels sequentially
- **Use case**: Testing, uniform distribution over time
- **Weights**: Equal cycling through all levels

```bash
--synthetic_request_generator_config_priority_distribution_type 1
```

### 2. UNIFORM (type=2)
- **Description**: Equal probability for each priority level
- **Use case**: Baseline testing, unbiased workloads
- **Weights** (5 levels): [0.20, 0.20, 0.20, 0.20, 0.20]

```bash
--synthetic_request_generator_config_priority_distribution_type 2
```

### 3. NORMAL (type=3)
- **Description**: Gaussian-like distribution centered on middle priority
- **Use case**: Most requests at normal priority, fewer at extremes
- **Weights** (5 levels): [0.05, 0.20, 0.50, 0.20, 0.05]
- **Expected distribution**: 5% critical, 20% high, 50% normal, 20% low, 5% background

```bash
--synthetic_request_generator_config_priority_distribution_type 3
```

### 4. POWER_LAW (type=4)
- **Description**: Heavy tail distribution - most requests at normal, few at critical
- **Use case**: Realistic workloads where high-priority requests are rare
- **Weights** (5 levels): [0.02, 0.08, 0.70, 0.15, 0.05]
- **Expected distribution**: 2% critical, 8% high, 70% normal, 15% low, 5% background

```bash
--synthetic_request_generator_config_priority_distribution_type 4
```

**Example output** (100 requests):
```
Priority 0 (critical):  1 request   (1%)
Priority 1 (high):     11 requests (11%)
Priority 2 (normal):   73 requests (73%)
Priority 3 (low):      13 requests (13%)
Priority 4 (background): 2 requests (2%)
```

### 5. ENTERPRISE (type=5)
- **Description**: Enterprise workload mix
- **Use case**: Business applications with clear priority tiers
- **Weights** (5 levels): [0.10, 0.30, 0.50, 0.08, 0.02]
- **Expected distribution**: 10% critical, 30% high, 50% normal, 8% low, 2% background

```bash
--synthetic_request_generator_config_priority_distribution_type 5
```

### 6. BURSTIER (type=6)
- **Description**: Bursty workload with occasional high-priority spikes
- **Use case**: Systems with periodic critical operations
- **Weights** (5 levels): [0.10, 0.20, 0.60, 0.08, 0.02]
- **Expected distribution**: 10% critical, 20% high, 60% normal, 8% low, 2% background

```bash
--synthetic_request_generator_config_priority_distribution_type 6
```

### 7. TIME_OF_DAY (type=7)
- **Description**: Time-varying distribution simulating daily cycles
- **Use case**: Workloads that vary by time (e.g., business hours vs. night)
- **Behavior**: 
  - Peak hours (40-60% of simulation cycle): More high-priority requests (ENTERPRISE mix)
  - Off-peak hours: More background requests (TRAFFIC_CLASS mix)

```bash
--synthetic_request_generator_config_priority_distribution_type 7
```

### 8. TRAFFIC_CLASS (type=8)
- **Description**: Web traffic pattern - mostly background with some urgent requests
- **Use case**: Public-facing services, batch processing workloads
- **Weights** (5 levels): [0.02, 0.08, 0.15, 0.20, 0.55]
- **Expected distribution**: 2% critical, 8% high, 15% normal, 20% low, 55% background

```bash
--synthetic_request_generator_config_priority_distribution_type 8
```

## Custom Weights

You can also specify custom weights for each priority level:

```bash
--synthetic_request_generator_config_priority_weights "[0.15, 0.25, 0.40, 0.15, 0.05]"
```

Weights must sum to approximately 1.0 and have length equal to `num_priority_levels`.

## Chrome Trace Visualization

When `--metrics_config_enable_chrome_trace` is enabled, priority information is visible in:

1. **Request Arrival Events**: Shows priority when request enters system
   - Name: `Request {id} Arrival (P{priority})`
   - Category: `request_lifecycle`
   - Args include: `request_id`, `priority`, `num_prefill_tokens`, `num_decode_tokens`

2. **Dispatch Events**: Shows priority when request is assigned to replica
   - Name: `Dispatch Req {id} to Replica {rid} (P{priority})`
   - Category: `scheduling`
   - Args include: `request_id`, `priority`, `replica_id`

3. **Batch Events**: Shows priorities of all requests in batch
   - Includes `request_priorities` array and overall `batch_priority`

## Llumnix Scheduler Behavior

The Llumnix global scheduler respects priorities when placing requests:

1. **Priority-aware placement**: Groups requests by priority, schedules highest priority first
2. **Freeness-based selection**: Among non-draining replicas, selects one with highest freeness
3. **Headroom management**: Local schedulers reserve headroom blocks for high-priority requests
4. **Virtual usage**: Priority headroom is included in virtual usage calculation

## Example Scenarios

### Test priority ordering under moderate load:
```bash
python3 -m vidur.main \
  --global_scheduler_config_type llumnix \
  --cluster_config_num_replicas 3 \
  --synthetic_request_generator_config_num_requests 100 \
  --synthetic_request_generator_config_num_priority_levels 5 \
  --synthetic_request_generator_config_priority_distribution_type 3 \
  --poisson_request_interval_generator_config_qps 5 \
  --time_limit 30 \
  --metrics_config_enable_chrome_trace
```

### Stress test with enterprise workload:
```bash
python3 -m vidur.main \
  --global_scheduler_config_type llumnix \
  --cluster_config_num_replicas 4 \
  --synthetic_request_generator_config_num_requests 500 \
  --synthetic_request_generator_config_num_priority_levels 5 \
  --synthetic_request_generator_config_priority_distribution_type 5 \
  --poisson_request_interval_generator_config_qps 20 \
  --time_limit 60 \
  --metrics_config_enable_chrome_trace \
  --llumnix_global_scheduler_config_enable_migration
```

### Verify priority distribution:
```bash
# Run simulation
python3 -m vidur.main \
  --synthetic_request_generator_config_num_priority_levels 5 \
  --synthetic_request_generator_config_priority_distribution_type 4 \
  --metrics_config_enable_chrome_trace \
  [other options...]

# Check distribution in Chrome trace
jq -r '.traceEvents[] | select(.name | contains("Arrival")) | .args.priority' \
  simulator_output/*/chrome_trace.json | sort | uniq -c
```

## Implementation Details

- **Priority assignment**: Occurs in `SyntheticRequestGenerator._generate_next_request()`
- **Sampler**: `vidur.utils.priority_sampler.PrioritySampler`
- **Distribution types**: `vidur.types.priority_distribution_type.PriorityDistributionType`
- **Config fields**: `SyntheticRequestGeneratorConfig` in `vidur/config/config.py`

## Related Configuration

- **Priority headroom**: `--llumlet_scheduler_config_priority_headroom_blocks` (default: 2400)
- **High priority threshold**: `--llumlet_scheduler_config_high_priority_threshold` (default: 1)
  - Priorities ≤ threshold receive headroom
- **Autoscale bands**: Affect when replicas drain (impacts priority execution)
  - `--llumnix_global_scheduler_config_autoscale_low` (default: -0.5)
  - `--llumnix_global_scheduler_config_autoscale_high` (default: 1.5)
