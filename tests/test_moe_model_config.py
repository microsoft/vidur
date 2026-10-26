"""Tests for MoE model configuration additions to VIDUR."""

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
        assert cfg.qk_rope_head_dim == 64

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
