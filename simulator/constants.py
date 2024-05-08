import os

PY_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_FILE = f"{PY_ROOT_DIR}/config/default.yml"
MODEL_CONFIG_DIR = f"{PY_ROOT_DIR}/../data/model_configs"
DEVICE_CONFIG_DIR = f"{PY_ROOT_DIR}/../data/device_configs"
CACHE_DIR = f"{PY_ROOT_DIR}/../.simulator_cache"

LOGGER_FORMAT = (
    "[%(asctime)s][%(filename)s:%(lineno)d:%(funcName)s][%(levelname)s] %(message)s"
)
LOGGER_TIME_FORMAT = "%H:%M:%S"
