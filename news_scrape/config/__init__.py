import logging
import logging.config
import os
from typing import Any

import yaml

LOGGING_CONFIG_FILE = "logging.yaml"

logging_config_path = os.path.join(os.path.dirname(__file__), LOGGING_CONFIG_FILE)

with open(file=logging_config_path, mode="r") as file:
    config: Any = yaml.safe_load(stream=file.read())
    logging.config.dictConfig(config)
