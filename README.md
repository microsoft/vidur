# LLM Inference Simulator

## Overview

LLM inference system simulator.

## Setup

To run the simulator, create a conda environment with the given dependency file.

```
conda env create -p ./env -f ./environment.yml
```

## Formatting Code

To run the code formatters execute the following command,

```
make format
```

## Setting up wandb

First, setup your account on https://microsoft-research.wandb.io/, obtain the api key and then run the following command,

```
wandb login --host https://microsoft-research.wandb.io
```

If you wish to skip wandb setup, simply comment out `wandb_project` and `wandb_group` in `simulator/config/default.yml`.

## Running simulator

To run the simulator, simply execute the following command from the repository root,

```
python -m simulator.main
```

The metrics will be logged to wandb directly and copy will be stored in `simulator_output` directory along with the chrome trace. Description of all the logged metrics can be found [here](docs/simulator_metrics.md).

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
