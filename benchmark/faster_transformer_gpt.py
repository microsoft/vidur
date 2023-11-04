import sys

import torch

try:
    sys.path.append("/repo/FasterTransformer")
    from examples.pytorch.gpt.utils.gpt_decoder import GptContextDecoder, GptDecoder, GptLayerWeights
    from examples.pytorch.gpt.utils.gpt import GptInitModelParameters
except ImportError:
    print("FasterTransformer disabled")

from benchmark.cuda_timer import CudaTimer

FT_LIB_PATH = "/repo/FasterTransformer/build/lib/libth_transformer.so"


class FasterTransformerBlock:
    def __init__(self, config) -> None:
        torch.classes.load_library(FT_LIB_PATH)
        self.gpt_layer_weights = GptLayerWeights.from_config(config)
        self.decoder = GptDecoder.from_config(config)
        self.context_decoder = GptContextDecoder.from_config(config)

    def init_weights(self):
        self.gpt_layer_weights.generate_random_weights(torch.float16, device="cuda")
        self.decoder.set_weight(self.gpt_layer_weights)
        self.context_decoder.set_weight(self.gpt_layer_weights)

    def forward(self, x, sequence_lengths, finished=None, total_padding_tokens=None, masked_tokens=None, attention_mask=None, k_cache=None, v_cache=None):
        with CudaTimer("overall"):
            if k_cache is None or v_cache is None:
                self.context_decoder.forward(
                    input_embeds=x,
                    attention_mask=attention_mask,
                    input_lengths=sequence_lengths,
                )
            else:
                input_length = k_cache.shape[4]
                output = self.decoder.forward(
                    max_input_length=input_length,
                    step=input_length + 1,
                    ite=0,
                    input_embeds=x,
                    sequence_lengths=sequence_lengths,
                    key_cache=k_cache,
                    value_cache=v_cache,
                    finished=finished,
                    total_padding_tokens=total_padding_tokens,
                    masked_tokens=masked_tokens,
                )
                # print(output)
                torch.cuda.synchronize()


def generate_faster_transformer_config(n_head, n_embd, max_seq_len):
    return GptInitModelParameters(
        head_num=n_head,
        size_per_head=(n_embd // n_head),
        layer_num=1,
        max_seq_len=max_seq_len,
        tensor_para_size=1,
        vocab_size=50257,
        start_id=50256,
        end_id=50256,
        pipeline_para_size=1,
        data_type="f16",
        weights_data_type="f16",
        layernorm_eps=1e-6,
        layernorm_type='pre_layernorm',
        activation_type='gelu',
        has_positional_encoding=True,
        has_pre_decoder_layernorm=False,
        has_post_decoder_layernorm=True,
        has_adapters=False,
        adapter_inter_size=0,
        int8_mode=0,
        sparse=False,
        is_free_buffer_after_forward=False,
    )
