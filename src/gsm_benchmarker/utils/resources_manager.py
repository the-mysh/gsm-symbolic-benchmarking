import logging
from importlib.resources import files
import json
from typing import Any
from pathlib import Path


logger = logging.getLogger(__name__)
_RESOURCES_PATH = files("gsm_benchmarker")/"resources"


def load_resource_json(file_name: str) -> dict[str, Any]:
    full_path = _RESOURCES_PATH / file_name
    logger.debug(f"Loading resource from file: {Path(full_path).resolve()}")
    data_bytes = full_path.read_bytes()
    data_dict = json.loads(data_bytes)
    return data_dict


def load_json_file(file_name: str | Path) -> dict[str, Any]:
    with open(file_name, "r") as f:
        data_dict = json.load(f)
    return data_dict
