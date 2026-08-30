"""
Generate TTFT (time-to-first-token) and TBT (time-between-tokens) plots
bucketed by request priority for Llumnix runs.

Data sources (auto-detected):
- TTFT: request_metrics.csv → prefill_e2e_time
- TBT preferred: plots/decode_token_interarrival_time_per_request.csv
- TBT fallback: request_metrics.csv → decode_time_execution_plus_preemption_normalized
- If neither exists, we fall back to batch durations from chrome_trace.json

Usage:
    python vidur/metrics/ttft_tbt_plots.py --run-dir <sim_output_dir>

If --run-dir is omitted, the most recent directory inside simulator_output/ is
used. Plots are written to <run-dir>/plots/.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

# Use a non-interactive backend for CLI / headless runs
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency at runtime
    wandb = None


def _wandb_image(path: Path, key: str) -> None:
    """Log an image to an active wandb run if available."""
    if wandb is None or not wandb.run:
        return
    wandb.log({key: wandb.Image(str(path))}, commit=False)


def _wandb_metrics(namespace: str, metrics: Dict[str, float]) -> None:
    """Log scalar metrics under a namespace to an active wandb run if available."""
    if wandb is None or not wandb.run or not metrics:
        return
    wandb.log({f"{namespace}/{k}": v for k, v in metrics.items()}, commit=False)


def _find_latest_run(sim_output_root: Path) -> Path:
    """Pick the newest directory inside simulator_output/."""
    run_dirs: List[Path] = [p for p in sim_output_root.iterdir() if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(
            f"No simulator outputs found under {sim_output_root}. "
            "Pass --run-dir explicitly if outputs live elsewhere."
        )
    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def _load_trace_events(trace_path: Path) -> List[dict]:
    with trace_path.open() as f:
        data = json.load(f)
    events = data.get("traceEvents", [])
    if not isinstance(events, list):
        raise ValueError(f"Unexpected chrome trace format in {trace_path}")
    return events


def _extract_request_priorities(trace_events: Iterable[dict]) -> Dict[int, int]:
    """
    Build request_id -> priority map from chrome trace events.
    Priority lives in args.request_priorities (one per request in the batch).
    """
    mapping: Dict[int, int] = {}
    for ev in trace_events:
        args = ev.get("args", {})

        # request_ids is required; skip if missing or empty
        req_ids = args.get("request_ids") or []
        if not req_ids:
            continue

        # retrieve per-request priorities if present
        req_prios = args.get("request_priorities") or []

        # Fallback: if priorities missing or all None, use batch_priority
        if (not req_prios) or all(p is None for p in req_prios):
            batch_prio = args.get("batch_priority")
            if batch_prio is not None:
                req_prios = [batch_prio for _ in req_ids]
            else:
                continue

        for req_id, prio in zip(req_ids, req_prios):
            if req_id in mapping and mapping[req_id] != prio:
                # keep the first seen value and warn, but do not fail
                print(
                    f"[warn] Request {req_id} priority mismatch: "
                    f"{mapping[req_id]} vs {prio}. Using {mapping[req_id]}."
                )
                continue
            mapping[int(req_id)] = int(prio)
    return mapping


def _extract_tbt(trace_events: Iterable[dict]) -> pd.DataFrame:
    """
    Get per-batch execution durations (proxy for TBT) and associated priority.
    Duration is stored in microseconds in chrome trace, convert to seconds.
    """
    rows: List[Tuple[int, float]] = []
    for ev in trace_events:
        args = ev.get("args", {})
        prio = args.get("batch_priority")
        if prio is None:
            req_prios = args.get("request_priorities") or []
            if req_prios and len(set(req_prios)) == 1:
                prio = req_prios[0]
        if prio is None:
            continue

        dur_us = ev.get("dur")
        if dur_us is None:
            continue
        rows.append((int(prio), float(dur_us) / 1e6))

    return pd.DataFrame(rows, columns=["priority", "tbt_seconds"])


def _plot_cdf(
    df: pd.DataFrame,
    value_col: str,
    output_path: Path,
    title: str,
    xlabel: str,
) -> None:
    """Draw a simple CDF split by priority."""
    if df.empty:
        print(f"[warn] No data available for {title}; skipping plot.")
        return

    plt.figure(figsize=(8, 5))
    ax = sns.ecdfplot(data=df, x=value_col, hue="priority")
    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    plt.title(title)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(title="Priority")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[info] Wrote {output_path}")
    _wandb_image(output_path, f"plots/{output_path.name}")
    _wandb_image(output_path, f"plots/{output_path.name}")
    _wandb_image(output_path, f"plots/{output_path.name}")
    _wandb_image(output_path, f"plots/{output_path.name}")


def _plot_hist(
    df: pd.DataFrame,
    value_col: str,
    output_path: Path,
    title: str,
    xlabel: str,
    log_x: bool = False,
) -> None:
    if df.empty:
        print(f"[warn] No data available for {title}; skipping plot.")
        return

    plt.figure(figsize=(8, 5))
    ax = sns.histplot(
        data=df, x=value_col, hue="priority", bins=30, element="step", stat="density"
    )
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(title="Priority")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[info] Wrote {output_path}")
    _wandb_image(output_path, f"plots/{output_path.name}")
    _wandb_image(output_path, f"plots/{output_path.name}")
    _wandb_image(output_path, f"plots/{output_path.name}")
    _wandb_image(output_path, f"plots/{output_path.name}")


def _plot_box_violin(
    df: pd.DataFrame,
    value_col: str,
    output_box: Path,
    output_violin: Path,
    title_prefix: str,
    xlabel: str,
) -> None:
    if df.empty:
        print(f"[warn] No data available for {title_prefix}; skipping box/violin.")
        return

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="priority", y=value_col)
    plt.xlabel("Priority")
    plt.ylabel(xlabel)
    plt.title(f"{title_prefix} (box)")
    plt.tight_layout()
    output_box.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_box)
    plt.close()
    print(f"[info] Wrote {output_box}")
    _wandb_image(output_box, f"plots/{output_box.name}")

    plt.figure(figsize=(8, 5))
    sns.violinplot(data=df, x="priority", y=value_col, cut=0, scale="width")
    plt.xlabel("Priority")
    plt.ylabel(xlabel)
    plt.title(f"{title_prefix} (violin)")
    plt.tight_layout()
    plt.savefig(output_violin)
    plt.close()
    print(f"[info] Wrote {output_violin}")
    _wandb_image(output_violin, f"plots/{output_violin.name}")


def _plot_timeseries(
    df: pd.DataFrame,
    value_col: str,
    output_path: Path,
    title: str,
    xlabel: str,
    x_col: str,
) -> None:
    if df.empty:
        print(f"[warn] No data available for {title}; skipping time series.")
        return

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x=x_col, y=value_col, hue="priority", marker="o", linewidth=1)
    plt.xlabel(xlabel)
    plt.ylabel(value_col)
    plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[info] Wrote {output_path}")


def _plot_scatter(
    df: pd.DataFrame,
    output_path: Path,
    title: str,
    x_col: str,
    y_col: str,
) -> None:
    if df.empty:
        print(f"[warn] No data available for {title}; skipping scatter.")
        return

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue="priority", alpha=0.7)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[info] Wrote {output_path}")


def _plot_bar_summary(
    df: pd.DataFrame,
    value_col: str,
    output_path: Path,
    title: str,
    metrics: Optional[List[str]] = None,
) -> None:
    if df.empty:
        print(f"[warn] No data available for {title}; skipping bar chart.")
        return

    agg_map = {
        "mean": "mean",
        "p50": lambda s: s.quantile(0.5),
        "p95": lambda s: s.quantile(0.95),
        "p99": lambda s: s.quantile(0.99),
    }
    stats = df.groupby("priority")[value_col].agg(**agg_map).reset_index()

    if metrics is not None:
        stats = stats[["priority"] + metrics]

    melted = stats.melt(id_vars="priority", var_name="metric", value_name=value_col)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=melted, x="priority", y=value_col, hue="metric")
    plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[info] Wrote {output_path}")


def _plot_throughput_latency(
    df: pd.DataFrame,
    throughput_col: str,
    latency_col: str,
    output_path: Path,
    title: str,
) -> None:
    if df.empty or throughput_col not in df.columns:
        print(f"[warn] No throughput data available for {title}; skipping.")
        return

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x=throughput_col, y=latency_col, hue="priority", alpha=0.7)
    plt.xlabel("Throughput (tokens/sec)")
    plt.ylabel(latency_col)
    plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[info] Wrote {output_path}")


def _print_stats(df: pd.DataFrame, value_col: str, label: str) -> None:
    if df.empty:
        print(f"[warn] No data for {label}; skipping stats.")
        return
    grouped = df.groupby("priority")[value_col]
    print(f"[info] {label} stats by priority:")
    for prio, series in grouped:
        desc = series.describe(percentiles=[0.5, 0.9, 0.95, 0.99])
        print(
            f"  prio {prio}: n={int(desc['count'])}, "
            f"mean={desc['mean']:.4f}, p50={desc['50%']:.4f}, "
            f"p90={desc['90%']:.4f}, p95={desc['95%']:.4f}, p99={desc['99%']:.4f}"
        )
        _wandb_metrics(
            f"stats/{label.lower()}",
            {
                f"prio_{prio}_count": float(desc["count"]),
                f"prio_{prio}_mean": float(desc["mean"]),
                f"prio_{prio}_p50": float(desc["50%"]),
                f"prio_{prio}_p90": float(desc["90%"]),
                f"prio_{prio}_p95": float(desc["95%"]),
                f"prio_{prio}_p99": float(desc["99%"]),
            },
        )


def _plot_distribution(
    df: pd.DataFrame,
    value_col: str,
    output_path: Path,
    title: str,
    xlabel: str,
) -> None:
    """Plot histogram + KDE split by priority."""
    if df.empty:
        print(f"[warn] No data available for {title}; skipping distribution plot.")
        return

    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x=value_col, hue="priority", kde=True, stat="density", common_norm=False)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[info] Wrote {output_path}")


def _plot_tail_summary(df: pd.DataFrame, value_col: str, output_path: Path, title: str) -> None:
    _plot_bar_summary(df, value_col, output_path, title, metrics=["p95", "p99"])


def _plot_bucket_violin(
    df: pd.DataFrame,
    value_col: str,
    output_path: Path,
    title: str,
    xlabel: str,
    buckets: Optional[List[float]] = None,
) -> None:
    if df.empty:
        print(f"[warn] No data available for {title}; skipping bucket violin.")
        return

    if buckets is None:
        buckets = [0.33, 0.66]

    low_q, high_q = buckets
    low = df[value_col].quantile(low_q)
    high = df[value_col].quantile(high_q)

    def _bucketize(v):
        if v <= low:
            return "low"
        if v <= high:
            return "normal"
        return "high"

    df = df.copy()
    df["latency_bucket"] = df[value_col].apply(_bucketize)

    plt.figure(figsize=(9, 5))
    sns.violinplot(data=df, x="latency_bucket", y=value_col, hue="priority", cut=0, scale="width")
    plt.xlabel("Latency bucket")
    plt.ylabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[info] Wrote {output_path}")


def _load_metric_df(request_df: pd.DataFrame, col: str, rename: Optional[str] = None) -> pd.DataFrame:
    if col not in request_df.columns:
        return pd.DataFrame()
    res = request_df[["Request Id", "priority", col]].dropna(subset=["priority", col])
    if rename:
        res = res.rename(columns={col: rename})
    return res


def _load_request_metrics(run_dir_path: Path, request_priorities: Dict[int, int]) -> pd.DataFrame:
    request_metrics_path = run_dir_path / "request_metrics.csv"
    if not request_metrics_path.exists():
        raise FileNotFoundError(
            f"request_metrics.csv not found in {run_dir_path}. "
            "Re-run the simulator with metrics enabled or pass a different --run-dir."
        )
    request_df = pd.read_csv(request_metrics_path)
    request_df["priority"] = request_df["Request Id"].map(request_priorities)
    # if priorities were missing in trace, default all to 0 so plots still work
    if request_df["priority"].isna().all():
        print("[warn] No per-request priorities found; defaulting to priority=0 for all requests.")
        request_df["priority"] = 0
    else:
        request_df["priority"] = request_df["priority"].fillna(0)
    return request_df


def _load_tbt(run_dir_path: Path, request_priorities: Dict[int, int], trace_events: List[dict], request_df: pd.DataFrame) -> pd.DataFrame:
    interarrival_path = run_dir_path / "plots" / "decode_token_interarrival_time_per_request.csv"
    if interarrival_path.exists():
        raw_tbt_df = pd.read_csv(interarrival_path)
        if "decode_token_interarrival_time" not in raw_tbt_df.columns:
            raise ValueError(
                f"Unexpected columns in {interarrival_path}, expected decode_token_interarrival_time."
            )
        tbt_df = raw_tbt_df[["Request Id", "decode_token_interarrival_time"]].rename(
            columns={"decode_token_interarrival_time": "tbt_seconds"}
        )
        tbt_df["priority"] = tbt_df["Request Id"].map(request_priorities)
    else:
        # Fallback to per-request averaged decode time.
        tbt_df = request_df[
            ["Request Id", "priority", "decode_time_execution_plus_preemption_normalized"]
        ].rename(
            columns={"decode_time_execution_plus_preemption_normalized": "tbt_seconds"}
        )
        if tbt_df["tbt_seconds"].dropna().empty:
            # last resort: batch durations from trace
            tbt_df = _extract_tbt(trace_events)
    tbt_df["priority"] = tbt_df["priority"].fillna(0)
    tbt_df = tbt_df.dropna(subset=["tbt_seconds", "priority"])
    if tbt_df.empty:
        raise ValueError(
            "No TBT data found. Ensure token inter-arrival or decode timings are logged."
        )
    return tbt_df


def _load_ttft(request_df: pd.DataFrame) -> pd.DataFrame:
    ttft_df = request_df[["Request Id", "priority", "prefill_e2e_time"]].dropna(
        subset=["priority", "prefill_e2e_time"]
    )
    if ttft_df.empty:
        raise ValueError("No TTFT data after joining priorities with request metrics.")
    return ttft_df


def _build_time_axis(request_df: pd.DataFrame) -> pd.Series:
    if "request_inter_arrival_delay" in request_df.columns:
        # reconstruct arrival timestamps cumulatively
        arrivals = request_df["request_inter_arrival_delay"].fillna(0).cumsum()
        return arrivals
    # fallback: request order
    return pd.Series(range(len(request_df)))


def _synthetic_demo(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create synthetic TTFT/TBT data with natural-looking distributions.
    """
    rng = np.random.default_rng(7)
    num_requests = 300
    priorities = rng.integers(0, 3, size=num_requests)

    # TTFT per priority: lognormal with small shifts
    ttft_base = {0: 0.18, 1: 0.22, 2: 0.25}
    ttft = np.array([rng.lognormal(mean=np.log(ttft_base[p]), sigma=0.1) for p in priorities])

    # TBT per priority: lognormal; higher priority slightly faster
    tbt_base = {0: 0.012, 1: 0.014, 2: 0.017}
    tbt = np.array([rng.lognormal(mean=np.log(tbt_base[p]), sigma=0.15) for p in priorities])

    # Prefill and decode components for demos
    prefill = np.array([rng.lognormal(mean=np.log(ttft_base[p] * 0.6), sigma=0.1) for p in priorities])
    decode = tbt * rng.integers(50, 120, size=num_requests) / 100  # decode duration scaled

    req_ids = np.arange(num_requests)
    request_df = pd.DataFrame(
        {
            "Request Id": req_ids,
            "priority": priorities,
            "prefill_e2e_time": ttft,
            "request_e2e_time": prefill + decode,
            "decode_time_execution_plus_preemption_normalized": tbt,
            "request_inter_arrival_delay": rng.exponential(scale=0.2, size=num_requests),
            "prefill_time_execution_plus_preemption": prefill,
        }
    )

    ttft_df = request_df[["Request Id", "priority", "prefill_e2e_time"]]
    tbt_df = request_df[["Request Id", "priority", "decode_time_execution_plus_preemption_normalized"]].rename(
        columns={"decode_time_execution_plus_preemption_normalized": "tbt_seconds"}
    )

    # ensure output dir exists for plots
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    return ttft_df, tbt_df, request_df


def _load_run_dir(run_dir: Optional[Path]) -> Path:
    if run_dir is not None:
        return run_dir
    sim_output_root = Path("simulator_output")
    return _find_latest_run(sim_output_root)


def main(run_dir: Optional[str] = None, demo: bool = False) -> None:
    run_dir_path = _load_run_dir(Path(run_dir) if run_dir else None) if not demo else Path(run_dir or "simulator_output/demo")
    plots_dir = run_dir_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if demo:
        print("[info] Running in demo mode with synthetic data.")
        ttft_df, tbt_df, request_df = _synthetic_demo(run_dir_path)
        trace_events = []
    else:
        print(f"[info] Using run directory: {run_dir_path}")
        chrome_trace_path = run_dir_path / "chrome_trace.json"
        if not chrome_trace_path.exists():
            raise FileNotFoundError(
                f"chrome_trace.json not found in {run_dir_path}. "
                "Enable chrome tracing in metrics_config to generate it."
            )
        trace_events = _load_trace_events(chrome_trace_path)
        request_priorities = _extract_request_priorities(trace_events)
        if not request_priorities:
            print("[warn] No per-request priorities found; using batch_priority fallback.")
        request_df = _load_request_metrics(run_dir_path, request_priorities)
        ttft_df = _load_ttft(request_df)
        tbt_df = _load_tbt(run_dir_path, request_priorities, trace_events, request_df)

    # Additional metric slices
    e2e_df = _load_metric_df(request_df, "request_e2e_time")
    prefill_df = _load_metric_df(request_df, "prefill_e2e_time")
    decode_df = _load_metric_df(
        request_df,
        "decode_time_execution_plus_preemption_normalized",
        rename="decode_seconds",
    )

    # Histogram / KDE
    _plot_hist(
        ttft_df,
        value_col="prefill_e2e_time",
        output_path=plots_dir / "ttft_hist.png",
        title="TTFT Histogram",
        xlabel="Seconds",
    )
    _plot_hist(
        tbt_df,
        value_col="tbt_seconds",
        output_path=plots_dir / "tbt_hist.png",
        title="TBT Histogram",
        xlabel="Seconds / token",
        log_x=True,
    )
    _plot_hist(
        prefill_df,
        value_col="prefill_e2e_time",
        output_path=plots_dir / "prefill_hist.png",
        title="Prefill Latency Histogram",
        xlabel="Seconds",
    )
    _plot_hist(
        decode_df,
        value_col="decode_seconds",
        output_path=plots_dir / "decode_hist.png",
        title="Decode Latency Histogram",
        xlabel="Seconds / token",
        log_x=True,
    )
    _plot_hist(
        e2e_df,
        value_col="request_e2e_time",
        output_path=plots_dir / "e2e_hist.png",
        title="E2E Latency Histogram",
        xlabel="Seconds",
    )

    # Box / Violin
    _plot_box_violin(
        ttft_df,
        value_col="prefill_e2e_time",
        output_box=plots_dir / "ttft_box.png",
        output_violin=plots_dir / "ttft_violin.png",
        title_prefix="TTFT by Priority",
        xlabel="Seconds",
    )
    _plot_box_violin(
        tbt_df,
        value_col="tbt_seconds",
        output_box=plots_dir / "tbt_box.png",
        output_violin=plots_dir / "tbt_violin.png",
        title_prefix="TBT by Priority",
        xlabel="Seconds / token",
    )
    _plot_box_violin(
        prefill_df,
        value_col="prefill_e2e_time",
        output_box=plots_dir / "prefill_box.png",
        output_violin=plots_dir / "prefill_violin.png",
        title_prefix="Prefill by Priority",
        xlabel="Seconds",
    )
    _plot_box_violin(
        decode_df,
        value_col="decode_seconds",
        output_box=plots_dir / "decode_box.png",
        output_violin=plots_dir / "decode_violin.png",
        title_prefix="Decode by Priority",
        xlabel="Seconds / token",
    )
    _plot_box_violin(
        e2e_df,
        value_col="request_e2e_time",
        output_box=plots_dir / "e2e_box.png",
        output_violin=plots_dir / "e2e_violin.png",
        title_prefix="E2E by Priority",
        xlabel="Seconds",
    )

    # Time series (request order or reconstructed arrival)
    request_df = request_df.sort_values(by="Request Id")
    request_df["arrival_or_idx"] = _build_time_axis(request_df)
    ttft_ts = request_df[["priority", "arrival_or_idx", "prefill_e2e_time"]].dropna(subset=["priority", "prefill_e2e_time"])
    tbt_req_mean = tbt_df.groupby("Request Id")["tbt_seconds"].mean().reset_index()
    tbt_ts = tbt_req_mean.merge(request_df[["Request Id", "priority", "arrival_or_idx"]], on="Request Id", how="left").dropna(subset=["priority", "tbt_seconds"])

    _plot_timeseries(
        ttft_ts,
        value_col="prefill_e2e_time",
        output_path=plots_dir / "ttft_timeseries.png",
        title="TTFT over time",
        xlabel="Arrival / Request index",
        x_col="arrival_or_idx",
    )
    _plot_timeseries(
        tbt_ts,
        value_col="tbt_seconds",
        output_path=plots_dir / "tbt_timeseries.png",
        title="TBT over time",
        xlabel="Arrival / Request index",
        x_col="arrival_or_idx",
    )

    # Scatter TTFT vs TBT
    ttft_tbt = ttft_df.merge(
        tbt_req_mean.rename(columns={"tbt_seconds": "tbt_mean"}),
        on="Request Id",
        how="inner",
    )
    _plot_scatter(
        ttft_tbt,
        output_path=plots_dir / "ttft_vs_tbt_scatter.png",
        title="TTFT vs TBT (per request)",
        x_col="prefill_e2e_time",
        y_col="tbt_mean",
    )

    # CDFs
    _plot_cdf(
        ttft_df,
        value_col="prefill_e2e_time",
        output_path=plots_dir / "ttft_cdf.png",
        title="TTFT CDF",
        xlabel="Seconds",
    )
    _plot_cdf(
        tbt_df,
        value_col="tbt_seconds",
        output_path=plots_dir / "tbt_cdf.png",
        title="TBT CDF",
        xlabel="Seconds / token",
    )

    # Bar summaries
    _plot_bar_summary(
        ttft_df,
        value_col="prefill_e2e_time",
        output_path=plots_dir / "ttft_bar_summary.png",
        title="TTFT summary by priority",
    )
    _plot_bar_summary(
        tbt_df,
        value_col="tbt_seconds",
        output_path=plots_dir / "tbt_bar_summary.png",
        title="TBT summary by priority",
    )
    _plot_bar_summary(
        prefill_df,
        value_col="prefill_e2e_time",
        output_path=plots_dir / "prefill_bar_summary.png",
        title="Prefill summary by priority",
    )
    _plot_bar_summary(
        decode_df,
        value_col="decode_seconds",
        output_path=plots_dir / "decode_bar_summary.png",
        title="Decode summary by priority",
    )
    _plot_bar_summary(
        e2e_df,
        value_col="request_e2e_time",
        output_path=plots_dir / "e2e_bar_summary.png",
        title="E2E summary by priority",
    )

    # Tail latency summaries (p95/p99)
    _plot_tail_summary(
        ttft_df,
        value_col="prefill_e2e_time",
        output_path=plots_dir / "ttft_tail_summary.png",
        title="TTFT tail (p95/p99) by priority",
    )
    _plot_tail_summary(
        tbt_df,
        value_col="tbt_seconds",
        output_path=plots_dir / "tbt_tail_summary.png",
        title="TBT tail (p95/p99) by priority",
    )
    _plot_tail_summary(
        prefill_df,
        value_col="prefill_e2e_time",
        output_path=plots_dir / "prefill_tail_summary.png",
        title="Prefill tail (p95/p99) by priority",
    )
    _plot_tail_summary(
        decode_df,
        value_col="decode_seconds",
        output_path=plots_dir / "decode_tail_summary.png",
        title="Decode tail (p95/p99) by priority",
    )
    _plot_tail_summary(
        e2e_df,
        value_col="request_e2e_time",
        output_path=plots_dir / "e2e_tail_summary.png",
        title="E2E tail (p95/p99) by priority",
    )

    # Throughput vs latency (tokens/sec derived from TBT)
    safe_tbt = tbt_df["tbt_seconds"].replace(0, np.nan)
    tbt_df["throughput_tokens_per_sec"] = 1.0 / safe_tbt
    _plot_throughput_latency(
        tbt_df,
        throughput_col="throughput_tokens_per_sec",
        latency_col="tbt_seconds",
        output_path=plots_dir / "throughput_vs_tbt.png",
        title="Throughput vs TBT",
    )

    # Throughput vs p99 latency per priority (aggregated)
    agg_throughput = (
        tbt_df.groupby("priority")
        .agg(
            throughput_tokens_per_sec=("throughput_tokens_per_sec", "mean"),
            p99_tbt=("tbt_seconds", lambda s: s.quantile(0.99)),
        )
        .reset_index()
    )
    _plot_scatter(
        agg_throughput,
        output_path=plots_dir / "throughput_vs_p99_tbt.png",
        title="Throughput vs p99 TBT by priority",
        x_col="throughput_tokens_per_sec",
        y_col="p99_tbt",
    )

    # Extra distribution plot
    _plot_distribution(
        ttft_df,
        value_col="prefill_e2e_time",
        output_path=plots_dir / "ttft_distribution_by_priority.png",
        title="TTFT Distribution by Priority",
        xlabel="Seconds",
    )
    _plot_distribution(
        tbt_df,
        value_col="tbt_seconds",
        output_path=plots_dir / "tbt_distribution_by_priority.png",
        title="TBT Distribution by Priority",
        xlabel="Seconds",
    )
    _plot_distribution(
        prefill_df,
        value_col="prefill_e2e_time",
        output_path=plots_dir / "prefill_distribution_by_priority.png",
        title="Prefill Distribution by Priority",
        xlabel="Seconds",
    )
    _plot_distribution(
        decode_df,
        value_col="decode_seconds",
        output_path=plots_dir / "decode_distribution_by_priority.png",
        title="Decode Distribution by Priority",
        xlabel="Seconds / token",
    )
    _plot_distribution(
        e2e_df,
        value_col="request_e2e_time",
        output_path=plots_dir / "e2e_distribution_by_priority.png",
        title="E2E Distribution by Priority",
        xlabel="Seconds",
    )

    # Latency buckets (low/normal/high) per metric
    _plot_bucket_violin(
        ttft_df,
        value_col="prefill_e2e_time",
        output_path=plots_dir / "ttft_bucket_violin.png",
        title="TTFT buckets by priority",
        xlabel="Seconds",
    )
    _plot_bucket_violin(
        tbt_df,
        value_col="tbt_seconds",
        output_path=plots_dir / "tbt_bucket_violin.png",
        title="TBT buckets by priority",
        xlabel="Seconds / token",
    )
    _plot_bucket_violin(
        prefill_df,
        value_col="prefill_e2e_time",
        output_path=plots_dir / "prefill_bucket_violin.png",
        title="Prefill buckets by priority",
        xlabel="Seconds",
    )
    _plot_bucket_violin(
        decode_df,
        value_col="decode_seconds",
        output_path=plots_dir / "decode_bucket_violin.png",
        title="Decode buckets by priority",
        xlabel="Seconds / token",
    )
    _plot_bucket_violin(
        e2e_df,
        value_col="request_e2e_time",
        output_path=plots_dir / "e2e_bucket_violin.png",
        title="E2E buckets by priority",
        xlabel="Seconds",
    )

    _print_stats(ttft_df, "prefill_e2e_time", "TTFT")
    _print_stats(tbt_df, "tbt_seconds", "TBT")
    _print_stats(prefill_df, "prefill_e2e_time", "Prefill")
    _print_stats(decode_df, "decode_seconds", "Decode")
    _print_stats(e2e_df, "request_e2e_time", "E2E")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot TTFT and TBT grouped by priority level for Llumnix runs."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Simulator output directory containing request_metrics.csv and chrome_trace.json. "
        "Defaults to the newest directory under simulator_output/.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate plots from synthetic data (no simulator output required).",
    )
    args = parser.parse_args()
    main(args.run_dir, demo=args.demo)
