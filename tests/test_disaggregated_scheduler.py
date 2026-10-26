import pytest

from vidur.config.model_config import BaseModelConfig
from vidur.scheduler.disaggregated_scheduler import (
    DisaggregatedScheduler,
    kv_cache_bytes_per_token,
)


def make_scheduler(
    num_prefill_replicas=1,
    num_decode_replicas=1,
    prefill_time_fn=lambda num_tokens: 0.0,
    decode_step_time_fn=lambda batch_size, avg_kv_tokens: 0.01,
    kv_bytes_per_token=0,
    kv_transfer_bandwidth_gb_per_s=1e9,
    max_decode_batch_size=128,
):
    return DisaggregatedScheduler(
        num_prefill_replicas=num_prefill_replicas,
        num_decode_replicas=num_decode_replicas,
        prefill_time_fn=prefill_time_fn,
        decode_step_time_fn=decode_step_time_fn,
        kv_bytes_per_token=kv_bytes_per_token,
        kv_transfer_bandwidth_gb_per_s=kv_transfer_bandwidth_gb_per_s,
        max_decode_batch_size=max_decode_batch_size,
    )


class TestKVCacheBytesPerToken:
    def test_dense_gqa(self):
        model_config = BaseModelConfig.create_from_name("meta-llama/Meta-Llama-3-8B")
        head_dim = model_config.embedding_dim // model_config.num_q_heads
        expected = (
            2 * model_config.num_kv_heads * head_dim * model_config.num_layers * 2
        )
        assert kv_cache_bytes_per_token(model_config) == expected

    def test_mla_uses_latent_dim(self):
        model_config = BaseModelConfig.create_from_name("deepseek-ai/DeepSeek-V3")
        # (kv_lora_rank + qk_rope_head_dim) * num_layers * fp16
        assert kv_cache_bytes_per_token(model_config) == (512 + 64) * 61 * 2

    def test_mla_is_much_smaller_than_per_head_kv(self):
        model_config = BaseModelConfig.create_from_name("deepseek-ai/DeepSeek-V3")
        head_dim = model_config.embedding_dim // model_config.num_q_heads
        per_head_kv = (
            2 * model_config.num_kv_heads * head_dim * model_config.num_layers * 2
        )
        assert per_head_kv / kv_cache_bytes_per_token(model_config) > 20


class TestDisaggregatedScheduler:
    def test_single_request_timing_is_exact(self):
        scheduler = make_scheduler(
            prefill_time_fn=lambda num_tokens: num_tokens * 1e-3,
            decode_step_time_fn=lambda batch_size, avg_kv_tokens: 0.01,
            kv_bytes_per_token=10**6,
            kv_transfer_bandwidth_gb_per_s=1.0,
        )
        result = scheduler.simulate([(0.0, 100, 11)])

        assert result.ttft_s["mean"] == pytest.approx(0.1)
        assert result.kv_transfer_s["mean"] == pytest.approx(0.1)
        # first token comes from prefill, the other 10 from decode iterations
        assert result.e2e_s["mean"] == pytest.approx(0.1 + 0.1 + 10 * 0.01)
        assert result.tpot_s["mean"] == pytest.approx((0.1 + 10 * 0.01) / 10)

    def test_continuous_batching_shares_decode_iterations(self):
        requests = [(0.0, 10, 11)] * 8

        batched = make_scheduler(max_decode_batch_size=8).simulate(requests)
        unbatched = make_scheduler(max_decode_batch_size=1).simulate(requests)

        # the first request starts decoding alone, the other 7 join at the next
        # iteration boundary and all finish one iteration after it
        assert batched.makespan_s == pytest.approx(11 * 0.01)
        assert batched.max_decode_batch_size_observed == 8
        assert unbatched.makespan_s == pytest.approx(8 * 10 * 0.01)

    def test_decode_batch_size_cap_is_respected(self):
        requests = [(0.0, 10, 50)] * 20
        result = make_scheduler(max_decode_batch_size=4).simulate(requests)

        assert result.max_decode_batch_size_observed == 4

    def test_decode_step_time_sees_batch_size_and_kv_length(self):
        seen = []

        def decode_step_time_fn(batch_size, avg_kv_tokens):
            seen.append((batch_size, avg_kv_tokens))
            return 0.01

        make_scheduler(decode_step_time_fn=decode_step_time_fn).simulate(
            [(0.0, 100, 3), (0.0, 300, 3)]
        )

        # the second request joins at the first iteration boundary; each running
        # request's kv grows by one token per iteration
        assert seen == [(1, 101), (2, 201), (1, 302)]

    def test_kv_transfers_serialise_on_prefill_link(self):
        scheduler = make_scheduler(
            prefill_time_fn=lambda num_tokens: 0.01,
            kv_bytes_per_token=10**6,
            kv_transfer_bandwidth_gb_per_s=1.0,
        )
        # each transfer takes 0.1 s, longer than the 0.01 s prefill
        result = scheduler.simulate([(0.0, 100, 2)] * 3)

        # transfer time includes waiting for the link: 0.1, 0.19, 0.28
        assert result.kv_transfer_s["mean"] == pytest.approx((0.1 + 0.19 + 0.28) / 3)

    def test_more_prefill_replicas_reduce_ttft(self):
        requests = [(i * 0.01, 512, 16) for i in range(50)]

        def run(num_prefill_replicas):
            return make_scheduler(
                num_prefill_replicas=num_prefill_replicas,
                prefill_time_fn=lambda num_tokens: num_tokens * 1e-4,
            ).simulate(requests)

        assert run(4).ttft_s["p90"] < run(1).ttft_s["p90"]

    def test_single_output_token_skips_transfer_and_decode(self):
        result = make_scheduler(prefill_time_fn=lambda num_tokens: 0.05).simulate(
            [(0.0, 100, 1)]
        )

        assert result.e2e_s["mean"] == pytest.approx(0.05)
        assert result.kv_transfer_s["mean"] == 0.0
        assert result.mean_decode_batch_size == 0.0
