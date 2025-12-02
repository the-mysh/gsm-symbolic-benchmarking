import logging
from importlib.resources import files
import json
from typing import Any
from pathlib import Path
import sys
import importlib.util
import os
import inspect


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


def load_8shot_functions(file_name: str):
    target_file_path = str(_RESOURCES_PATH / file_name)

    # 1. Dynamic Import Setup
    module_dir = os.path.dirname(target_file_path)
    module_name = os.path.splitext(os.path.basename(target_file_path))[0]
    sys.path.insert(0, module_dir) # Add directory to path

    external_module = None
    try:
        # Use importlib to load the module dynamically
        spec = importlib.util.spec_from_file_location(module_name, target_file_path)
        if spec:
            external_module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = external_module
            spec.loader.exec_module(external_module)
        else:
            raise ImportError(f"Could not load specification for {target_file_path}")

    except Exception as e:
        raise RuntimeError(f"Error loading module: {e}")

    finally:
        # Crucial: Clean up the path
        if module_dir in sys.path:
            sys.path.remove(module_dir)

    if not external_module:
        raise RuntimeError(f"Failed to load external module '{target_file_path}'")

    # 2. Extract and Inspect the Functions List
    function_sources = []

    if external_module:
        try:
            # Access the list of function objects directly from the imported module
            function_list = external_module.FUNCTIONS

            logger.debug(f"Found {len(function_list)} functions in the 'FUNCTIONS' list.")

            for func in function_list:
                # Check if the item is actually a function before inspecting
                if inspect.isfunction(func):
                    function_sources.append(inspect.getsource(func))  # add source code of the function to the list

        except AttributeError:
            raise RuntimeError(f"Error: The 'FUNCTIONS' list was not found in {module_name}.")

        except TypeError as e:
            raise RuntimeError(f"Error getting source code: {e}")

    return function_sources