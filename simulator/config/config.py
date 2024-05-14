import argparse
import datetime
import logging
import os

import yaml

from simulator.constants import DEFAULT_CONFIG_FILE, DEVICE_CONFIG_DIR, MODEL_CONFIG_DIR

logger = logging.getLogger(__name__)


class Config:
    def __init__(self, config_file=DEFAULT_CONFIG_FILE):
        self._parser = argparse.ArgumentParser()
        self._args = None
        self._load_yaml(config_file)
        self._parse_args()
        self._add_derived_args()
        self._write_yaml_to_file()
        logger.info(f"Config: {self.get_yaml()}")

    def _load_yaml(self, filename):
        with open(filename, "r") as file:
            yaml_config = yaml.safe_load(file)
        self._update_namespace(yaml_config)

    def _parse_args(self):
        self._args = self._parser.parse_args()

    def _add_derived_args(self):
        self._args.output_dir = f"{self._args.output_dir}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}"
        os.makedirs(self._args.output_dir, exist_ok=True)
        self._load_model_config()
        self._load_device_config()
        self._substitute_variables_in_args()

    def _update_namespace(self, config_dict, parent_key=""):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                new_key = f"{parent_key}{key}_" if parent_key else f"{key}_"
                self._update_namespace(value, new_key)
            else:
                arg_name = f"{parent_key}{key}"

                if type(value) == bool:
                    self._parser.add_argument(
                        f"--{arg_name}",
                        default=value,
                        action=argparse.BooleanOptionalAction,
                    )
                elif arg_name in [
                    "simulator_time_limit",
                    "metrics_store_subsamples",
                    "replica_scheduler_num_blocks",
                ]:
                    self._parser.add_argument(f"--{arg_name}", default=value, type=int)
                else:
                    self._parser.add_argument(
                        f"--{arg_name}", default=value, type=type(value)
                    )

    def __getattr__(self, name):
        return getattr(self._args, name, None)

    def get_yaml(self):
        return yaml.dump(self._args.__dict__, default_flow_style=False)

    def _write_yaml_to_file(self):
        with open(f"{self._args.output_dir}/config.yml", "w") as file:
            file.write(self.get_yaml())

    def to_dict(self):
        return self._args.__dict__

    def _add_to_args(self, new_args_dict, parent_key=""):
        for key, value in new_args_dict.items():
            arg_name = f"{parent_key}{key}"
            setattr(self._args, arg_name, value)

    def _load_model_config(self):
        assert self.replica_model_name is not None

        config_file = f"{MODEL_CONFIG_DIR}/{self.replica_model_name}.yml"
        with open(config_file, "r") as file:
            yaml_config = yaml.safe_load(file)
        self._add_to_args(yaml_config, "replica_")

    def _load_device_config(self):
        assert self.replica_device is not None

        config_file = f"{DEVICE_CONFIG_DIR}/{self.replica_device}.yml"
        with open(config_file, "r") as file:
            yaml_config = yaml.safe_load(file)
        self._add_to_args(yaml_config, "replica_")

    def _substitute_variables_in_args(self):
        assert self.replica_model_name is not None
        assert self.replica_device is not None
        assert self.replica_network_device is not None

        # update names of sklearn config files
        for key, value in self._args.__dict__.items():
            if isinstance(value, str):
                self._args.__dict__[key] = (
                    value.replace("{MODEL}", self.replica_model_name)
                    .replace("{DEVICE}", self.replica_device)
                    .replace("{NETWORK_DEVICE}", self.replica_network_device)
                )
