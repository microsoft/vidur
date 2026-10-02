"""Compare failover reports from several simulation runs.

    python -m vidur.failover.compare --output_dir out \\
        no_failure=out/no_failure lumen=out/lumen

Each run directory is the --metrics_config_output_dir of a simulation (the
newest timestamped run inside it is used) or a run directory itself. Runs are
expected to replay the same requests (same seed and workload), so latencies of
the interrupted and affected requests can be compared by request id, also for
runs without failures.
"""

import argparse
import glob
import json
import os
from typing import List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from vidur.failover.failover_manager import summarize_latencies

REPORT_FILE = "failover_report.json"
REQUESTS_FILE = "failover_request_metrics.csv"
TIMELINE_FILE = "failover_output_tokens_per_second.csv"

Run = Tuple[str, str, dict, pd.DataFrame]


def find_run_dir(path: str) -> str:
    if os.path.exists(os.path.join(path, REPORT_FILE)):
        return path

    run_dirs = sorted(
        os.path.dirname(report)
        for report in glob.glob(os.path.join(path, "*", REPORT_FILE))
    )
    if not run_dirs:
        raise FileNotFoundError(f"No {REPORT_FILE} found under {path}")
    return run_dirs[-1]


def get_request_ids(runs: List[Run], group: str) -> Optional[Set[int]]:
    """Requests the latency columns cover, taken from the runs with failures."""
    if group == "all":
        return None

    request_ids = set()
    for _, _, report, requests in runs:
        request_ids |= set(requests.loc[requests["interrupted"], "request_id"])
        if group != "affected":
            continue
        for failure in report["failures"]:
            end = failure["recovered_at"]
            arrived = requests["arrived_at"].between(
                failure["failed_at"], float("inf") if end is None else end
            )
            request_ids |= set(requests.loc[arrived, "request_id"])
    return request_ids


def summarize(label: str, report: dict, requests: pd.DataFrame) -> dict:
    latencies = summarize_latencies(requests)
    failures = report["failures"]
    row = {
        "run": label,
        "requests": latencies["num_requests"],
        "mean_ttft_s": latencies["ttft"]["mean"],
        "p99_ttft_s": latencies["ttft"]["p99"],
        "mean_tpot_ms": latencies["tpot"]["mean"] * 1e3,
        "p99_tpot_ms": latencies["tpot"]["p99"] * 1e3,
        "recovery_time_s": None,
        "mean_request_recovery_s": None,
        "interrupted": None,
        "restored_from_checkpoint": None,
        "redirected": None,
        "recomputed_tokens": None,
    }
    if failures:
        recovery_times = [f["recovery_time"] for f in failures]
        row["recovery_time_s"] = None if None in recovery_times else max(recovery_times)
        latencies = [
            (f["mean_request_recovery_latency"], f["num_interrupted_requests"])
            for f in failures
            if f["mean_request_recovery_latency"] is not None
        ]
        if latencies:
            row["mean_request_recovery_s"] = sum(m * n for m, n in latencies) / sum(
                n for _, n in latencies
            )
        row["interrupted"] = sum(f["num_interrupted_requests"] for f in failures)
        row["restored_from_checkpoint"] = sum(
            f["num_restored_from_checkpoint"] for f in failures
        )
        row["redirected"] = sum(f["num_redirected"] for f in failures)
        row["recomputed_tokens"] = sum(f["num_recomputed_tokens"] for f in failures)
    return row


def to_markdown(df: pd.DataFrame) -> str:
    def fmt(value) -> str:
        if value is None or pd.isna(value):
            return "-"
        if isinstance(value, float) and not value.is_integer():
            return f"{value:.2f}"
        return str(int(value)) if isinstance(value, float) else str(value)

    lines = [
        "| " + " | ".join(df.columns) + " |",
        "|" + "---|" * len(df.columns),
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def plot_timeline(
    runs: List[Run], output_path: str, smoothing_s: int, margin_s: float
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    failures = []
    for label, run_dir, report, _ in runs:
        timeline = pd.read_csv(os.path.join(run_dir, TIMELINE_FILE))
        if timeline.empty:
            continue
        # fill seconds without output so the rolling mean is over wall time
        seconds = range(int(timeline["time_s"].max()) + 1)
        series = timeline.set_index("time_s")["output_tokens"].reindex(
            seconds, fill_value=0
        )
        ax.plot(
            series.index,
            series.rolling(smoothing_s, center=True, min_periods=1).mean(),
            label=label,
        )
        failures.extend(report["failures"])

    for failed_at, recovered_at in {
        (f["failed_at"], f["recovered_at"]) for f in failures
    }:
        ax.axvline(failed_at, color="black", linestyle="--", linewidth=0.8)
        if recovered_at is not None:
            ax.axvline(recovered_at, color="grey", linestyle=":", linewidth=0.8)

    if failures and margin_s > 0:
        start = min(f["failed_at"] for f in failures)
        end = max(f["recovered_at"] or f["failed_at"] for f in failures)
        ax.set_xlim(start - margin_s, end + 2 * margin_s)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Output tokens/s ({smoothing_s}s mean)")
    ax.set_title("Cluster output throughput (dashed: failure, dotted: reload done)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="label=path pairs")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--group",
        default="all",
        choices=["all", "interrupted", "affected"],
        help="Requests the latency columns cover: all, the interrupted ones, or "
        "those interrupted or arriving between a failure and the reload.",
    )
    parser.add_argument("--smoothing_s", type=int, default=5)
    parser.add_argument(
        "--margin_s",
        type=float,
        default=60,
        help="Plot from this long before the first failure to twice as long after "
        "the last reload; 0 plots the whole run.",
    )
    args = parser.parse_args()

    runs = []
    for run in args.runs:
        label, _, path = run.partition("=")
        if not path:
            label, path = os.path.basename(os.path.normpath(run)), run
        run_dir = find_run_dir(path)
        with open(os.path.join(run_dir, REPORT_FILE)) as f:
            report = json.load(f)
        requests = pd.read_csv(os.path.join(run_dir, REQUESTS_FILE))
        runs.append((label, run_dir, report, requests))

    request_ids = get_request_ids(runs, args.group)
    rows = []
    for label, _, report, requests in runs:
        if request_ids is not None:
            requests = requests[requests["request_id"].isin(request_ids)]
        rows.append(summarize(label, report, requests))
    df = pd.DataFrame(rows)
    table = to_markdown(df)
    print(table)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        df.to_csv(os.path.join(args.output_dir, "comparison.csv"), index=False)
        with open(os.path.join(args.output_dir, "comparison.md"), "w") as f:
            f.write(table + "\n")
        plot_timeline(
            runs,
            os.path.join(args.output_dir, "output_throughput.png"),
            args.smoothing_s,
            args.margin_s,
        )
        print(
            f"Wrote comparison.csv, comparison.md and output_throughput.png to {args.output_dir}"
        )


if __name__ == "__main__":
    main()
