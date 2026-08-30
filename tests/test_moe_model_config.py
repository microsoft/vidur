"""Tests for MoE model configuration additions to VIDUR."""

import os
import sys

import pytest

# Import model_config directly to avoid pulling in the full vidur stack
# (which requires sklearn, ray, etc.).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vidur.config.model_config import (
    BaseModelConfig,
    BaseMoEModelConfig,
    DeepSeekV3ModelConfig,
    MixtralModelConfig,
)


class TestDeepSeekV3ModelConfig:
    def test_name(self):
        assert DeepSeekV3ModelConfig.get_name() == "deepseek-ai/DeepSeek-V3"

    def test_moe_fields(self):
        cfg = DeepSeekV3ModelConfig(
            num_q_heads=128,
            num_kv_heads=128,
            embedding_dim=7168,
            mlp_hidden_dim=18432,
        )
        assert cfg.num_experts == 256
        assert cfg.num_active_experts == 8
        assert cfg.expert_intermediate_dim == 2048
        assert cfg.num_shared_experts == 1

    def test_mla_fields(self):
        cfg = DeepSeekV3ModelConfig(
            num_q_heads=128,
            num_kv_heads=128,
            embedding_dim=7168,
            mlp_hidden_dim=18432,
        )
        assert cfg.kv_lora_rank == 512
        assert cfg.q_lora_rank == 1536

    def test_is_moe_subclass(self):
        assert issubclass(DeepSeekV3ModelConfig, BaseMoEModelConfig)
        assert issubclass(DeepSeekV3ModelConfig, BaseModelConfig)

    def test_auto_discovery(self):
        """BaseFixedConfig.create_from_name should find DeepSeekV3 automatically."""
        found = BaseModelConfig.create_from_name("deepseek-ai/DeepSeek-V3")
        assert isinstance(found, DeepSeekV3ModelConfig)


class TestMixtralModelConfig:
    def test_name(self):
        assert MixtralModelConfig.get_name() == "mistralai/Mixtral-8x7B-v0.1"

    def test_moe_fields(self):
        cfg = MixtralModelConfig(
            num_q_heads=32,
            num_kv_heads=8,
            embedding_dim=4096,
            mlp_hidden_dim=14336,
        )
        assert cfg.num_experts == 8
        assert cfg.num_active_experts == 2
        assert cfg.num_shared_experts == 0

    def test_no_mla(self):
        cfg = MixtralModelConfig(
            num_q_heads=32,
            num_kv_heads=8,
            embedding_dim=4096,
            mlp_hidden_dim=14336,
        )
        assert cfg.kv_lora_rank == 0  # standard MHA

    def test_auto_discovery(self):
        found = BaseModelConfig.create_from_name("mistralai/Mixtral-8x7B-v0.1")
        assert isinstance(found, MixtralModelConfig)


class TestDisaggregatedScheduler:
    def test_kv_bytes_per_token(self):
        """Sanity check KV cache size formula."""
        from unittest.mock import MagicMock

        mod = self._load_scheduler_module()
        kv_cache_bytes_per_token = mod.kv_cache_bytes_per_token
        DisaggregatedScheduler = mod.DisaggregatedScheduler

        model_cfg = MagicMock()
        model_cfg.embedding_dim = 4096
        model_cfg.num_q_heads = 32
        model_cfg.num_kv_heads = 8
        model_cfg.num_layers = 32

        replica_cfg = MagicMock()
        replica_cfg.model_config = model_cfg

        head_dim = 4096 // 32  # 128
        expected = 2 * 8 * 128 * 32 * 2  # 2 * kv_heads * head_dim * layers * fp16
        assert kv_cache_bytes_per_token(replica_cfg) == expected

    def _load_scheduler_module(self):
        import importlib.util

        mod_name = "vidur.scheduler.disaggregated_scheduler"
        if mod_name in sys.modules:
            return sys.modules[mod_name]
        spec = importlib.util.spec_from_file_location(
            mod_name,
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "vidur",
                "scheduler",
                "disaggregated_scheduler.py",
            ),
        )
        mod = importlib.util.module_from_spec(spec)
        # Register before exec so dataclasses can resolve the module's annotations
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod

    def test_simulation_runs(self):
        """Run a small simulation end-to-end."""
        from unittest.mock import MagicMock

        mod = self._load_scheduler_module()
        DisaggregatedScheduler = mod.DisaggregatedScheduler

        model_cfg = MagicMock()
        model_cfg.embedding_dim = 4096
        model_cfg.num_q_heads = 32
        model_cfg.num_kv_heads = 8
        model_cfg.num_layers = 32

        replica_cfg = MagicMock()
        replica_cfg.model_config = model_cfg

        scheduler = DisaggregatedScheduler(
            replica_config=replica_cfg,
            prefill_fleet_size=2,
            decode_fleet_size=4,
            interconnect_bandwidth_gbps=400.0,
            prefill_time_fn=lambda seq_len: seq_len * 0.001,  # 1ms per token
            decode_time_fn=lambda seq_len, out_len: out_len * 0.02,  # 20ms per token
        )

        requests = [(i * 0.1, 512, 128) for i in range(20)]  # 20 requests
        result = scheduler.simulate(requests)

        assert result.num_requests == 20
        assert result.prefill_fleet_size == 2
        assert result.decode_fleet_size == 4
        assert result.p50_e2e_latency_s > 0
        assert result.throughput_rps > 0
        assert 0.0 <= result.prefill_utilisation <= 1.0
        assert 0.0 <= result.decode_utilisation <= 1.0

    def test_p_d_ratio_affects_ttft(self):
        """More prefill workers → lower TTFT."""
        from unittest.mock import MagicMock

        mod = self._load_scheduler_module()
        DisaggregatedScheduler = mod.DisaggregatedScheduler

        model_cfg = MagicMock()
        model_cfg.embedding_dim = 4096
        model_cfg.num_q_heads = 32
        model_cfg.num_kv_heads = 8
        model_cfg.num_layers = 32

        replica_cfg = MagicMock()
        replica_cfg.model_config = model_cfg

        requests = [(i * 0.05, 256, 64) for i in range(30)]

        def make_scheduler(n_prefill, n_decode):
            return DisaggregatedScheduler(
                replica_config=replica_cfg,
                prefill_fleet_size=n_prefill,
                decode_fleet_size=n_decode,
                interconnect_bandwidth_gbps=400.0,
                prefill_time_fn=lambda seq_len: seq_len * 0.002,
                decode_time_fn=lambda seq_len, out_len: out_len * 0.01,
            )

        result_1pf = make_scheduler(1, 5).simulate(requests)
        result_3pf = make_scheduler(3, 3).simulate(requests)

        # More prefill workers should reduce mean TTFT (or at worst not increase it)
        assert result_3pf.mean_ttft_s <= result_1pf.mean_ttft_s + 0.01
