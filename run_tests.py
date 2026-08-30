"""
Run Llumnix/Llumlet plots or Llumnix vs LOR+vLLM metric comparisons.

Two modes:
  - plots: generate latency plots for Llumnix+Llumlet only (existing behavior).
  - compare: run matched scenarios for Llumnix+Llumlet and LOR+vLLM, then compute
             aggregate metrics + speedups via vidur.metrics.system_metrics.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import wandb

from vidur.metrics.latency_config import LATENCY_TESTS_BY_SYSTEM, TEST_SCENARIO_MATRIX
from vidur.metrics import latency_analysis as la
from vidur.metrics import system_metrics as sm

SYSTEM_LLUMNIX = "llumnix_llumlet"
SYSTEM_LOR = "lor_vllm"


def _run_command(cmd: str) -> None:
    print(f"[info] Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def _latest_dir(root: Path) -> Path:
    candidates = [p for p in root.glob("*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No simulator outputs found under {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _find_new_run_dir(root: Path, before: set[Path]) -> Path:
    after = {p for p in root.glob("*") if p.is_dir()}
    new_dirs = after - before
    if new_dirs:
        return _latest_dir(Path(root))
    # fallback: no new dir detected; pick latest
    return _latest_dir(Path(root))


def _build_summary(run_dir: Path) -> Dict[str, float]:
    """Compute simple aggregates for wandb logging."""
    chrome_trace_path = run_dir / "chrome_trace.json"
    trace_events = la._load_trace_events(chrome_trace_path)
    priorities = la._extract_request_priorities(trace_events)
    request_df = la._load_request_metrics(run_dir, priorities)
    ttft_df = la._load_ttft(request_df)
    tbt_df = la._load_tbt(run_dir, priorities, trace_events, request_df)

    def qstats(series: pd.Series, prefix: str) -> Dict[str, float]:
        s = series.dropna()
        if s.empty:
            return {}
        return {
            f"{prefix}_mean": float(s.mean()),
            f"{prefix}_p50": float(s.quantile(0.50)),
            f"{prefix}_p90": float(s.quantile(0.90)),
            f"{prefix}_p95": float(s.quantile(0.95)),
            f"{prefix}_p99": float(s.quantile(0.99)),
        }

    summary = {}
    summary.update(qstats(ttft_df["prefill_e2e_time"], "ttft"))
    summary.update(qstats(tbt_df["tbt_seconds"], "tbt"))
    summary.update(qstats(request_df["request_e2e_time"], "e2e"))
    return summary


def _log_to_wandb(
    run,
    test_name: str,
    description: str,
    cmd: str,
    run_dir: Path,
    plots: List[Path],
    summary: Dict[str, float],
    step: int,
) -> None:
    if run is None:
        return
    images = [wandb.Image(str(p), caption=p.name) for p in plots]
    log_payload = {
        f"{test_name}/description": description,
        f"{test_name}/command": cmd,
        f"{test_name}/run_dir": str(run_dir),
        f"{test_name}/plots": images,
    }
    for k, v in summary.items():
        log_payload[f"{test_name}/{k}"] = v
    wandb.log(log_payload, step=step)


def _load_wandb_api_key(env_path: Path = Path(".env")) -> Optional[str]:
    """
    Read WANDB_API_KEY from a .env-style file.
    Keeps dependencies minimal (no python-dotenv requirement).
    """
    if not env_path.exists():
        return None
    key = None
    with env_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("WANDB_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    return key or None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Llumnix plots or Llumnix vs LOR+vLLM comparisons."
    )
    parser.add_argument(
        "--mode",
        choices=["plots", "compare"],
        default="plots",
        help="plots: Llumnix-only plots. compare: run Llumnix+Llumlet vs LOR+vLLM comparisons.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Optional scenario index (0-based) to run a single scenario.",
    )
    parser.add_argument(
        "--latency-target",
        type=float,
        default=None,
        help="Override latency target when computing cost-vs-latency metrics.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip latency plot generation (mostly useful in compare mode).",
    )
    return parser.parse_args()


def _select_tests(tests: List[dict], index: Optional[int]) -> List[dict]:
    if index is None:
        return tests
    if index < 0 or index >= len(tests):
        raise IndexError(f"Index {index} out of range for {len(tests)} tests.")
    return [tests[index]]


def _extract_flag_value(cmd: str, flag: str) -> Optional[str]:
    """Return the value following a CLI flag inside a command string."""
    tokens = cmd.split()
    for i, tok in enumerate(tokens):
        if tok == flag and i + 1 < len(tokens):
            return tokens[i + 1]
    return None


def _derive_compare_run_name(llumnix_cmd: str) -> str:
    """Build wandb run name like comparison_qps_X_req_Y from the Llumnix command."""
    qps = _extract_flag_value(
        llumnix_cmd, "--poisson_request_interval_generator_config_qps"
    ) or "unknown"
    num_req = _extract_flag_value(
        llumnix_cmd, "--synthetic_request_generator_config_num_requests"
    ) or "unknown"
    # strip any trailing punctuation/commas if present
    qps_clean = str(qps).strip().strip(",")
    req_clean = str(num_req).strip().strip(",")
    return f"comparison_qps_{qps_clean}_req_{req_clean}"


def _execute_test(test: dict, generate_plots: bool, step: int, wandb_run=None) -> Path:
    name = test["name"]
    desc = test.get("description", "")
    base_root = Path("simulator_output") / name
    before_dirs = {p for p in base_root.glob("*") if p.is_dir()}
    base_root.mkdir(parents=True, exist_ok=True)

    cmd = f"{test['cmd']} --metrics_config_output_dir {base_root}"
    _run_command(cmd)

    run_dir = _find_new_run_dir(base_root, before_dirs)
    print(f"[info] Latest run dir for {name}: {run_dir}")

    plots: List[Path] = []
    summary: Dict[str, float] = {}
    if generate_plots:
        la.main(str(run_dir))
        plots_dir = run_dir / "plots"
        plots = sorted(p for p in plots_dir.glob("*.png"))
        summary = _build_summary(run_dir)

    _log_to_wandb(
        wandb_run,
        test_name=name,
        description=desc,
        cmd=cmd,
        run_dir=run_dir,
        plots=plots,
        summary=summary,
        step=step,
    )

    return run_dir


def run_llumnix_plots(args: argparse.Namespace) -> None:
    api_key = _load_wandb_api_key()
    if api_key:
        wandb.login(key=api_key)

    tests = _select_tests(LATENCY_TESTS_BY_SYSTEM[SYSTEM_LLUMNIX], args.index)

    for idx, test in enumerate(tests):
        name = test["name"]
        desc = test.get("description", "")
        run_name = os.getenv("WANDB_RUN_NAME", name)

        wandb_run = wandb.init(
            project=os.getenv("WANDB_PROJECT", "llumnix-clean"),
            entity=os.getenv("WANDB_ENTITY"),
            mode=os.getenv("WANDB_MODE", "online"),
            name=run_name,
            group=os.getenv("WANDB_GROUP"),
            config={
                "test_name": name,
                "description": desc,
                "num_tests": len(tests),
                "system": SYSTEM_LLUMNIX,
            },
        )

        _execute_test(test, generate_plots=not args.skip_plots, step=idx, wandb_run=wandb_run)

        if wandb_run:
            wandb_run.finish()


def run_comparison(args: argparse.Namespace) -> None:
    api_key = _load_wandb_api_key()
    if api_key:
        wandb.login(key=api_key)

    scenario_items = sorted(TEST_SCENARIO_MATRIX.items())
    if args.index is not None:
        if args.index < 0 or args.index >= len(scenario_items):
            raise IndexError(f"Index {args.index} out of range for {len(scenario_items)} scenarios.")
        scenario_items = [scenario_items[args.index]]

    rows = []
    for step, (scenario_id, system_tests) in enumerate(scenario_items):
        if SYSTEM_LLUMNIX not in system_tests or SYSTEM_LOR not in system_tests:
            print(f"[warn] Skipping scenario {scenario_id} because one system is missing.")
            continue
        llumnix_test = system_tests[SYSTEM_LLUMNIX]
        lor_test = system_tests[SYSTEM_LOR]

        run_name = _derive_compare_run_name(llumnix_test["cmd"])
        wandb_run = wandb.init(
            project=os.getenv("WANDB_PROJECT", "llumnix-clean"),
            entity=os.getenv("WANDB_ENTITY"),
            mode=os.getenv("WANDB_MODE", "online"),
            name=run_name,
            group=os.getenv("WANDB_GROUP", "comparison"),
            config={
                "scenario": scenario_id,
                "llumnix_command": llumnix_test["cmd"],
                "lor_command": lor_test["cmd"],
            },
        )

        llumnix_run_dir = _execute_test(
            llumnix_test, generate_plots=False, step=step, wandb_run=wandb_run
        )
        lor_run_dir = _execute_test(
            lor_test, generate_plots=False, step=step, wandb_run=wandb_run
        )

        _, llumnix_metrics = sm.compute_run_metrics(
            llumnix_run_dir, SYSTEM_LLUMNIX, llumnix_test["name"], latency_target=args.latency_target
        )
        _, lor_metrics = sm.compute_run_metrics(
            lor_run_dir, SYSTEM_LOR, lor_test["name"], latency_target=args.latency_target
        )
        comparison = sm.compare_runs(llumnix_metrics, lor_metrics)

        rows.append(
            {
                "scenario": scenario_id,
                "llumnix_run_dir": str(llumnix_run_dir),
                "lor_run_dir": str(lor_run_dir),
                **comparison,
            }
        )

        print(f"[info] Scenario {scenario_id} speedups:")
        for metric, value in comparison.items():
            print(f"  {metric}: {value}")

        if wandb_run:
            payload = {
                "scenario": scenario_id,
                "llumnix_run_dir": str(llumnix_run_dir),
                "lor_run_dir": str(lor_run_dir),
            }
            payload.update({k: v for k, v in comparison.items() if v is not None})
            wandb.log(payload, step=step)
            wandb_run.finish()

    if rows:
        df = pd.DataFrame(rows)
        output_path = Path("simulator_output") / "comparison_metrics.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[info] Wrote comparison metrics to {output_path}")


if __name__ == "__main__":
    args = _parse_args()
    if args.mode == "compare":
        run_comparison(args)
    else:
        run_llumnix_plots(args)
