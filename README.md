# VIDUR: LLM Inference Simulator

## Setup

### Using `mamba`

To run the simulator, create a mamba environment with the given dependency file.

```sh
mamba env create -p ./env -f ./environment.yml
```

### Using `venv`

1. Ensure that you have python 3.10 installed on your system. Refer <https://www.bitecode.dev/p/installing-python-the-bare-minimum>
2. `cd` into the repository root
3. Create a virtual environment using `venv` module using `python3.10 -m venv .venv`
4. Activate the virtual environment using `source .venv/bin/activate`
5. Install the dependencies using `python -m pip install -r requirements.txt`
6. Run `deactivate` to deactivate the virtual environment

### Using `conda` (Least recommended)

To run the simulator, create a conda environment with the given dependency file.

```sh
conda env create -p ./env -f ./environment.yml
```

## Setting up wandb (Optional)

First, setup your account on <https://microsoft-research.wandb.io/>, obtain the api key and then run the following command,

```sh
wandb login --host https://microsoft-research.wandb.io
```

If you wish to skip wandb setup, simply comment out `wandb_project` and `wandb_group` in `simulator/config/default.yml`.

## Running simulator

To run the simulator, simply execute the following command from the repository root,

```sh
python -m simulator.main
```

or a big example with all the parameters,

```sh
python -m simulator.main \
--replica_model_name codellama/CodeLlama-34b-Instruct-hf \
--replica_num_layers 48 \
--replica_num_q_heads 64 \
--replica_num_kv_heads 8 \
--replica_embedding_dim 8192 \
--replica_mlp_hidden_dim 22016 \
--replica_vocab_size 32768 \
--replica_use_gated_mlp \
--replica_fp16_tflops 312 \
--replica_total_memory_gb 80 \
--sklearn_execution_time_predictor_compute_input_file ./data/profiling/a100/mlp.csv \
--sklearn_execution_time_predictor_attention_input_file ./data/profiling/a100/mixed_attention.csv \
--sklearn_execution_time_predictor_all_reduce_input_file ./data/profiling/a100/all_reduce.csv \ --sklearn_execution_time_predictor_send_recv_input_file ./data/profiling/a100/p2p_intra_node.csv \
--sklearn_execution_time_predictor_cpu_overhead_input_file ./data/profiling/a100/cpu_overheads.csv \
--cluster_num_replicas 1 \
--replica_num_tensor_parallel_workers 1 \
--request_generator_provider synthetic \
--request_generator_request_length_generator_provider trace \
--trace_request_length_generator_trace_file ./data/processed_traces/cnn_dailymail_stats_llama2_tokenizer.csv \
--request_generator_request_interval_generator_provider poisson \
--poisson_request_interval_generator_qps 0.75 \
--synthetic_request_generator_num_requests 256 \
--replica_scheduler_provider vllm \
--replica_scheduler_batch_size_cap 128
```

The simulator supports a plethora of parameters for the simulation description of which can be found [here](docs/simulator_params.md).

The metrics will be logged to wandb directly and copy will be stored in `simulator_output` directory along with the chrome trace. Description of all the logged metrics can be found [here](docs/simulator_metrics.md).

## Formatting Code

To run the code formatters execute the following command,

```sh
make format
```


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
