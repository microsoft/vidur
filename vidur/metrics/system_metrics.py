"""
Aggregate latency, efficiency, and cost metrics for Llumnix vs baseline systems.

This module mirrors the helper-driven style of `latency_analysis.py`: small
loader utilities plus pure functions that compute aggregate metrics. It is
purpose-built for comparing two scheduler stacks:
    - Llumnix (global) + Llumlet (replica)
    - INFaaS (global) + vLLM (replica)

Metrics computed per run:
    - end-to-end latency (mean, p99)
    - prefill latency (mean, p99)
    - decode latency (mean, p99)
    - preemption rate and preemption loss (share of e2e)
    - memory fragmentation (average, plus per-batch series)
    - resource usage (average instance count) and cost vs latency target
    - optional priority-aware slices (mean/p99 for highest-priority requests)

Comparison helpers then compute speedups (INFaaS→Llumnix) so results line up with
the Llumnix paper reporting style.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from vidur.metrics import latency_analysis as la

DEVICE_HOURLY_COST = {
    "a100": 3.0,
    "h100": 6.0,
}


@dataclass
class RunData:
    name: str
    system: str
    run_dir: Path
    config: Dict
    request_df: pd.DataFrame
    batch_df: pd.DataFrame


def _safe_quantile(series: pd.Series, q: float) -> Optional[float]:
    clean = series.dropna()
    if clean.empty:
        return None
    return float(clean.quantile(q))


def _safe_mean(series: pd.Series) -> Optional[float]:
    clean = series.dropna()
    if clean.empty:
        return None
    return float(clean.mean())


def _load_config(run_dir: Path) -> Dict:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {run_dir}")
    with config_path.open() as f:
        return json.load(f)


def _load_batch_metrics(run_dir: Path) -> pd.DataFrame:
    batch_path = run_dir / "batch_metrics.csv"
    if not batch_path.exists():
        return pd.DataFrame()
    return pd.read_csv(batch_path)


def _load_request_df(run_dir: Path) -> pd.DataFrame:
    chrome_trace_path = run_dir / "chrome_trace.json"
    trace_events = (
        la._load_trace_events(chrome_trace_path) if chrome_trace_path.exists() else []
    )
    request_priorities = (
        la._extract_request_priorities(trace_events) if trace_events else {}
    )
    return la._load_request_metrics(run_dir, request_priorities)


def _latency_stats(request_df: pd.DataFrame, column: str) -> Dict[str, Optional[float]]:
    if column not in request_df.columns:
        return {"mean": None, "p99": None}
    series = request_df[column]
    return {"mean": _safe_mean(series), "p99": _safe_quantile(series, 0.99)}


def _preemption_metrics(request_df: pd.DataFrame) -> Dict[str, Optional[float]]:
    col = "request_preemption_time"
    if col not in request_df.columns:
        return {"rate": None, "loss": None}

    preempt_times = request_df[col].fillna(0)
    total = len(preempt_times)
    rate = float((preempt_times > 0).sum() / total) if total else None

    if "request_e2e_time" in request_df.columns and total:
        loss = float(
            (preempt_times / request_df["request_e2e_time"].replace(0, np.nan))
            .replace([np.inf, -np.inf], np.nan)
            .mean()
        )
    else:
        loss = None
    return {"rate": rate, "loss": loss}


def _fragmentation_metrics(batch_df: pd.DataFrame, config: Dict) -> Dict[str, object]:
    if batch_df.empty:
        return {"avg_fragmentation": None, "series": pd.DataFrame()}

    sched_cfg = config.get("cluster_config", {}).get("replica_scheduler_config", {})
    block_size = sched_cfg.get("block_size") or config.get("cluster_config", {}).get(
        "replica_config", {}
    ).get("block_size")
    num_blocks = sched_cfg.get("num_blocks")

    if not block_size or not num_blocks:
        return {"avg_fragmentation": None, "series": pd.DataFrame()}

    capacity_tokens = block_size * num_blocks
    if "batch_num_tokens" not in batch_df.columns:
        return {"avg_fragmentation": None, "series": pd.DataFrame()}

    frag_series = 1.0 - (batch_df["batch_num_tokens"] / capacity_tokens)
    frag_series = frag_series.clip(lower=0.0, upper=1.0)
    series_df = pd.DataFrame(
        {
            "Batch Id": batch_df.get("Batch Id", range(len(batch_df))),
            "fragmentation": frag_series,
        }
    )
    return {"avg_fragmentation": _safe_mean(frag_series), "series": series_df}


def _resource_usage(
    config: Dict, latency_target: Optional[float]
) -> Dict[str, Optional[float]]:
    cluster_cfg = config.get("cluster_config", {})
    replica_cfg = cluster_cfg.get("replica_config", {})
    num_replicas = cluster_cfg.get("num_replicas") or 0
    device = (replica_cfg.get("device") or "").lower()
    cost_per_hour = DEVICE_HOURLY_COST.get(device)

    runtime_seconds = config.get("time_limit") or 0
    avg_instance_count = float(num_replicas)
    cost = None
    if runtime_seconds and cost_per_hour is not None:
        cost = float(avg_instance_count * (runtime_seconds / 3600.0) * cost_per_hour)

    cost_vs_latency = None
    if cost is not None and latency_target not in (None, 0):
        cost_vs_latency = float(cost / latency_target)

    return {
        "avg_instance_count": avg_instance_count,
        "runtime_seconds": float(runtime_seconds),
        "run_cost": cost,
        "cost_vs_latency_target": cost_vs_latency,
    }


def _priority_slice_metrics(
    request_df: pd.DataFrame, column: str
) -> Dict[str, Optional[float]]:
    if column not in request_df.columns or "priority" not in request_df.columns:
        return {"mean": None, "p99": None}
    high_prio = request_df["priority"].max()
    high_df = request_df[request_df["priority"] == high_prio]
    if high_df.empty:
        return {"mean": None, "p99": None}
    return {
        "mean": _safe_mean(high_df[column]),
        "p99": _safe_quantile(high_df[column], 0.99),
    }


def compute_run_metrics(
    run_dir: Path,
    system: str,
    name: Optional[str] = None,
    latency_target: Optional[float] = None,
) -> Tuple[RunData, Dict]:
    """Load a single run directory and compute aggregate metrics."""
    request_df = _load_request_df(run_dir)
    batch_df = _load_batch_metrics(run_dir)
    config = _load_config(run_dir)

    latency = _latency_stats(request_df, "request_e2e_time")
    prefill = _latency_stats(request_df, "prefill_e2e_time")
    decode = _latency_stats(
        request_df, "decode_time_execution_plus_preemption_normalized"
    )
    preemption = _preemption_metrics(request_df)
    fragmentation = _fragmentation_metrics(batch_df, config)
    resource = _resource_usage(config, latency_target or latency["p99"])
    priority_slice = _priority_slice_metrics(request_df, "request_e2e_time")

    metrics = {
        "latency": latency,
        "prefill": prefill,
        "decode": decode,
        "preemption": preemption,
        "fragmentation": {
            "avg": fragmentation["avg_fragmentation"],
            "series": fragmentation["series"],
        },
        "resource": resource,
        "priority": priority_slice,
    }

    return (
        RunData(
            name=name or run_dir.name,
            system=system,
            run_dir=run_dir,
            config=config,
            request_df=request_df,
            batch_df=batch_df,
        ),
        metrics,
    )


def _speedup(baseline: Optional[float], contender: Optional[float]) -> Optional[float]:
    if baseline is None or contender is None or contender == 0:
        return None
    return float(baseline / contender)


def compare_runs(
    llumnix_metrics: Dict, infaas_metrics: Dict
) -> Dict[str, Optional[float]]:
    """
    Compute speedups using INFaaS as baseline and Llumnix as contender.
    Speedup > 1.0 means Llumnix is faster.
    """

    return {
        "e2e_mean_speedup": _speedup(
            infaas_metrics["latency"]["mean"], llumnix_metrics["latency"]["mean"]
        ),
        "e2e_p99_speedup": _speedup(
            infaas_metrics["latency"]["p99"], llumnix_metrics["latency"]["p99"]
        ),
        "prefill_mean_speedup": _speedup(
            infaas_metrics["prefill"]["mean"], llumnix_metrics["prefill"]["mean"]
        ),
        "prefill_p99_speedup": _speedup(
            infaas_metrics["prefill"]["p99"], llumnix_metrics["prefill"]["p99"]
        ),
        "decode_mean_ratio": _speedup(
            infaas_metrics["decode"]["mean"], llumnix_metrics["decode"]["mean"]
        ),
        "decode_p99_ratio": _speedup(
            infaas_metrics["decode"]["p99"], llumnix_metrics["decode"]["p99"]
        ),
        "preemption_rate_delta": None
        if infaas_metrics["preemption"]["rate"] is None
        or llumnix_metrics["preemption"]["rate"] is None
        else float(
            infaas_metrics["preemption"]["rate"] - llumnix_metrics["preemption"]["rate"]
        ),
        "preemption_loss_delta": None
        if infaas_metrics["preemption"]["loss"] is None
        or llumnix_metrics["preemption"]["loss"] is None
        else float(
            infaas_metrics["preemption"]["loss"] - llumnix_metrics["preemption"]["loss"]
        ),
        "fragmentation_delta": None
        if infaas_metrics["fragmentation"]["avg"] is None
        or llumnix_metrics["fragmentation"]["avg"] is None
        else float(
            infaas_metrics["fragmentation"]["avg"]
            - llumnix_metrics["fragmentation"]["avg"]
        ),
        "cost_ratio": _speedup(
            infaas_metrics["resource"]["run_cost"],
            llumnix_metrics["resource"]["run_cost"],
        ),
        "cost_per_latency_ratio": _speedup(
            infaas_metrics["resource"]["cost_vs_latency_target"],
            llumnix_metrics["resource"]["cost_vs_latency_target"],
        ),
        "priority_mean_speedup": _speedup(
            infaas_metrics["priority"]["mean"], llumnix_metrics["priority"]["mean"]
        ),
        "priority_p99_speedup": _speedup(
            infaas_metrics["priority"]["p99"], llumnix_metrics["priority"]["p99"]
        ),
    }
