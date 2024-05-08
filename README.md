# Vidur: LLM Inference Simulator

We highly recommend to read the [paper](docs/vidur_paper.pdf) first before going into the code.

## Experiments

1. [Cross Model-Dataset Fidelity](experiments/REQUEST_MATCH.md) compares fidelity in terms for request e2e times between the simulator and vLLM for a wide variety of model, tensor parallel dimension and dataset combinations.
2. [LightLLM-Fidelity](experiments/LIGHTLLM_MATCH.md) compares fidelity in terms for request e2e times between the simulator and LightLLM for LLaMa-7B on datasets.
3. [Scheduler Match](experiments/SCHEDULER_MATCH.md) compares the simulator with different schedulers and compare the results with vLLM.

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

First, setup your account on `https://<your-org>.wandb.io/` or public wandb, obtain the api key and then run the following command,

```sh
wandb login --host https://<your-org>.wandb.io
```

To opt out of wandb, pick any one of the following methods:

1. `export WANDB_MODE=disabled` in your shell or add this in `~/.zshrc` or `~/.bashrc`. Remeber to reload using `source ~/.zshrc`.
2. Set `wandb_project` and `wandb_group` as `""` in `simulator/config/default.yml`. Also remove these CLI params from the shell command with which the simulator is invoked.

## Running simulator

To run the simulator, execute the following command from the repository root,

```sh
python -m simulator.main
```

or a big example with all the parameters,

```sh
python -m simulator.main  \
--replica_device a100 \
--replica_model_name meta-llama/Llama-2-7b-hf  \
--cluster_num_replicas 1 \
--replica_num_tensor_parallel_workers 1 \
--replica_num_pipeline_stages 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_length_provider trace \
--synthetic_request_generator_interval_provider static \
--request_generator_max_tokens 4096 \
--trace_request_length_generator_trace_file ./data/processed_traces/arxiv_summarization_filtered_stats_llama2_tokenizer.csv \
--synthetic_request_generator_num_requests 128  \
--request_generator_provider synthetic \
--synthetic_request_generator_length_provider trace \
--synthetic_request_generator_interval_provider static \
--request_generator_max_tokens 4096 \
--trace_request_length_generator_trace_file ./data/processed_traces/arxiv_summarization_filtered_stats_llama2_tokenizer.csv \
--synthetic_request_generator_num_requests 128  \
--replica_scheduler_provider vllm  \
--replica_scheduler_batch_size_cap 256  \
--vllm_scheduler_max_tokens_in_batch 4096
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
