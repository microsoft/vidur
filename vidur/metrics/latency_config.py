"""
Preset latency scenarios for two systems:
 - Llumnix (global) + Llumlet (replica)
 - INFaaS (global) + vLLM (replica)

Commands share identical workload knobs (arrival process, request lengths,
replica model/device, predictor settings, metrics config) to keep cross-system
comparisons fair. Only scheduler-specific flags differ between the two base
commands.
"""

from __future__ import annotations

from typing import Dict, List

WORKLOAD_BASE = [
    "python3 -m vidur.main",
    "--cluster_config_num_replicas 4",
    "--synthetic_request_generator_config_num_priority_levels 3",
    "--synthetic_request_generator_config_num_requests 2000",
    "--length_generator_config_type zipf",
    "--zipf_request_length_generator_config_max_tokens 512",
    "--zipf_request_length_generator_config_theta 1.2",
    "--zipf_request_length_generator_config_min_tokens 64",
    "--zipf_request_length_generator_config_prefill_to_decode_ratio 2.0",
    "--interval_generator_config_type poisson",
    "--poisson_request_interval_generator_config_qps 1250",
    "--replica_config_device a100",
    "--replica_config_model_name meta-llama/Llama-2-7b-hf",
    "--execution_time_predictor_config_type linear_regression",
    "--linear_regression_execution_time_predictor_config_prediction_max_batch_size 32",
    "--linear_regression_execution_time_predictor_config_prediction_max_tokens_per_request 8192",
    "--linear_regression_execution_time_predictor_config_no_cache",
    "--metrics_config_cache_dir /tmp/vidur_latency_no_cache",
    "--time_limit 60",
    "--metrics_config_enable_chrome_trace",
    "--metrics_config_write_metrics",
    "--metrics_config_store_request_metrics",
    "--log_level info",
]

SYSTEMS: Dict[str, Dict[str, object]] = {
    "llumnix_llumlet": {
        "slug": "llumnix_llumlet",
        "label": "Llumnix + Llumlet",
        "include_llumnix_priority": True,
        "base_command": WORKLOAD_BASE
        + [
            "--global_scheduler_config_type llumnix",
            "--llumnix_global_scheduler_config_num_priority_levels 3",
            "--llumnix_global_scheduler_config_enable_migration",
            "--llumnix_global_scheduler_config_rebalance_interval 0.05",
            "--replica_scheduler_config_type llumlet",
            "--llumlet_scheduler_config_num_blocks 128",
            "--llumlet_scheduler_config_block_size 16",
            "--llumlet_scheduler_config_batch_size_cap 64",
        ],
    },
    "infaas_vllm": {
        "slug": "infaas_vllm",
        "label": "INFaaS + vLLM",
        "include_llumnix_priority": False,
        "base_command": WORKLOAD_BASE
        + [
            "--global_scheduler_config_type infaas",
            "--infaas_global_scheduler_config_alpha 1.0",
            "--infaas_global_scheduler_config_beta 1.0",
            "--infaas_global_scheduler_config_gamma 1.0",
            "--infaas_global_scheduler_config_target_latency_ms 1000",
            "--infaas_global_scheduler_config_ewma_alpha 0.6",
            "--infaas_global_scheduler_config_overload_latency_factor 1.3",
            "--infaas_global_scheduler_config_interference_latency_factor 1.15",
            "--infaas_global_scheduler_config_queue_depth_threshold 2",
            "--infaas_global_scheduler_config_interference_queue_threshold 1",
            "--infaas_global_scheduler_config_overload_cooldown 3",
            "--infaas_global_scheduler_config_interference_cooldown 2",
            "--replica_scheduler_config_type vllm",
            "--vllm_scheduler_config_num_blocks 128",
            "--vllm_scheduler_config_block_size 16",
            "--vllm_scheduler_config_batch_size_cap 64",
            "--vllm_scheduler_config_max_tokens_in_batch 2048",
            "--vllm_scheduler_config_watermark_blocks_fraction 0.01",
        ],
    },
}


def cmd_with_overrides(system_key: str, *overrides: str) -> str:
    """Build a runnable CLI string for a specific system by appending overrides to its base command."""
    if system_key not in SYSTEMS:
        raise KeyError(f"Unknown system '{system_key}'. Known systems: {list(SYSTEMS)}")
    base_cmd = SYSTEMS[system_key]["base_command"]
    assert isinstance(base_cmd, list)
    return " ".join(base_cmd + list(overrides))


BASE_LATENCY_TESTS = [
    {
        "name": "baseline_migration_on",
        "description": "Baseline with migration enabled at nominal 100 QPS.",
        "overrides": {
            "llumnix_llumlet": [],
            "infaas_vllm": [],
        },
    },
    # Test Type 1: Migration & Load Balancing Sensitivity
    {
        "name": "migration_disabled",
        "description": "Migration disabled to evaluate imbalance and preemption without rescheduling.",
        "overrides": {
            "llumnix_llumlet": ["--no-llumnix_global_scheduler_config_enable_migration"],
            "infaas_vllm": [],
        },
    },
    {
        "name": "rebalance_aggressive",
        "description": "Aggressive rebalance interval to trigger frequent migrations and stress the scheduler.",
        "overrides": {
            "llumnix_llumlet": [
                "--llumnix_global_scheduler_config_rebalance_interval 0.01",
                "--llumnix_global_scheduler_config_load_imbalance_threshold 0.1",
            ],
            "infaas_vllm": [],
        },
    },
    # Test Type 2: KV Capacity & Fragmentation Stress
    {
        "name": "kv_capacity_tight",
        "description": "Tight KV capacity: 64 blocks and batch cap 16 to stress fragmentation and packing.",
        "overrides": {
            "llumnix_llumlet": [
                "--llumlet_scheduler_config_num_blocks 64",
                "--llumlet_scheduler_config_batch_size_cap 16",
            ],
            "infaas_vllm": [
                "--vllm_scheduler_config_num_blocks 64",
                "--vllm_scheduler_config_batch_size_cap 16",
            ],
        },
    },
]

PRIORITY_DISTRIBUTIONS = [
    # {"type": 1, "slug": "round_robin", "name": "ROUND_ROBIN"},
    {"type": 2, "slug": "uniform", "name": "UNIFORM"},
    {"type": 3, "slug": "normal", "name": "NORMAL"},
    {"type": 4, "slug": "power_law", "name": "POWER_LAW"},
    # {"type": 5, "slug": "enterprise", "name": "ENTERPRISE"},
    # {"type": 6, "slug": "burstier", "name": "BURSTIER"},
    # {"type": 7, "slug": "time_of_day", "name": "TIME_OF_DAY"},
    # {"type": 8, "slug": "traffic_class", "name": "TRAFFIC_CLASS"},
]

PRIORITY_LEVELS = [1, 2, 3, 4, 5]
REQUEST_COUNTS = [10000, 15000]


def _apply_priority_distribution(cmd: str, dist_type: int) -> str:
    """Ensure the command sets the requested priority distribution, removing any existing override."""
    tokens = cmd.split()
    filtered = []
    skip = False
    for tok in tokens:
        if skip:
            skip = False
            continue
        if tok == "--synthetic_request_generator_config_priority_distribution_type":
            skip = True
            continue
        filtered.append(tok)
    filtered.append(
        f"--synthetic_request_generator_config_priority_distribution_type {dist_type}"
    )
    return " ".join(filtered)


def _apply_priority_levels(cmd: str, num_levels: int, include_llumnix_flag: bool) -> str:
    """Ensure the command sets the requested number of priority levels for both Llumnix (if present) and generator."""
    tokens = cmd.split()
    filtered = []
    skip = False
    for tok in tokens:
        if skip:
            skip = False
            continue
        if tok == "--synthetic_request_generator_config_num_priority_levels":
            skip = True
            continue
        if (
            include_llumnix_flag
            and tok == "--llumnix_global_scheduler_config_num_priority_levels"
        ):
            skip = True
            continue
        filtered.append(tok)
    filtered.append(
        f"--synthetic_request_generator_config_num_priority_levels {num_levels}"
    )
    if include_llumnix_flag:
        filtered.append(
            f"--llumnix_global_scheduler_config_num_priority_levels {num_levels}"
        )
    return " ".join(filtered)


def _apply_num_requests(cmd: str, num_requests: int) -> str:
    """Ensure the command sets the requested number of synthetic requests."""
    tokens = cmd.split()
    filtered = []
    skip = False
    for tok in tokens:
        if skip:
            skip = False
            continue
        if tok == "--synthetic_request_generator_config_num_requests":
            skip = True
            continue
        filtered.append(tok)
    filtered.append(f"--synthetic_request_generator_config_num_requests {num_requests}")
    return " ".join(filtered)


def _expand_tests_with_distributions_levels_and_requests(base_tests, system_key: str):
    system = SYSTEMS[system_key]
    include_llumnix_flag = bool(system.get("include_llumnix_priority"))
    system_label = system["label"]
    system_slug = system["slug"]
    expanded = []
    for test in base_tests:
        overrides = test.get("overrides", {}).get(system_key, [])
        cmd = cmd_with_overrides(system_key, *overrides)
        for num_levels in PRIORITY_LEVELS:
            level_cmd = _apply_priority_levels(cmd, num_levels, include_llumnix_flag)
            for num_requests in REQUEST_COUNTS:
                req_cmd = _apply_num_requests(level_cmd, num_requests)
                for dist in PRIORITY_DISTRIBUTIONS:
                    dist_suffix = f"dist{dist['type']}_{dist['slug']}"
                    level_suffix = f"lvl{num_levels}"
                    req_suffix = f"req{num_requests}"
                    expanded.append(
                        {
                            "system": system_key,
                            "scenario": f"{test['name']}_{level_suffix}_{req_suffix}_{dist_suffix}",
                            "name": f"{system_slug}_{test['name']}_{level_suffix}_{req_suffix}_{dist_suffix}",
                            "description": (
                                f"{system_label}: {test['description']} "
                                f"Priority levels: {num_levels}. "
                                f"Requests: {num_requests}. "
                                f"Priority distribution: {dist['name']} (type={dist['type']})."
                            ),
                            "cmd": _apply_priority_distribution(req_cmd, dist["type"]),
                        }
                    )
    return expanded


LATENCY_TESTS_LLUMNIX = _expand_tests_with_distributions_levels_and_requests(
    BASE_LATENCY_TESTS, "llumnix_llumlet"
)
LATENCY_TESTS_INFAAS = _expand_tests_with_distributions_levels_and_requests(
    BASE_LATENCY_TESTS, "infaas_vllm"
)
LATENCY_TESTS_BY_SYSTEM: Dict[str, List[dict]] = {
    "llumnix_llumlet": LATENCY_TESTS_LLUMNIX,
    "infaas_vllm": LATENCY_TESTS_INFAAS,
}


def _pair_by_scenario(tests_by_system: Dict[str, List[dict]]) -> Dict[str, Dict[str, dict]]:
    """
    Build a mapping of scenario_id -> {system_key: test}.

    Used to align Llumnix and baseline runs for metric comparisons.
    """
    paired: Dict[str, Dict[str, dict]] = {}
    for system_key, tests in tests_by_system.items():
        for test in tests:
            scenario = test["scenario"]
            paired.setdefault(scenario, {})
            paired[scenario][system_key] = test
    return paired


TEST_SCENARIO_MATRIX = _pair_by_scenario(LATENCY_TESTS_BY_SYSTEM)

# Backwards compatibility: default to Llumnix-only suite
LATENCY_TESTS = LATENCY_TESTS_LLUMNIX
