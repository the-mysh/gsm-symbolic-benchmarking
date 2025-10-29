import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Any
import logging

from gsm_benchmarker.api_model_wrapper import APIType


logger = logging.getLogger(__name__)
_RESOURCES_PATH = files("gsm_benchmarker")/"resources"



@dataclass
class SingleModelConfig:
    name: str
    family: str
    size: float | None
    instruction_tuned: bool
    api_type: APIType | None
    extra_kwargs: dict[str, dict[str, Any]]

    @classmethod
    def from_json_dict(cls, d: dict):
        api_type_name = d.pop('api_type')
        if api_type_name is None:
            api_type = None
        else:
            api_type = APIType[api_type_name]

        try:
            size = float(d.pop('size'))
        except ValueError:
            size = None

        return cls(**d, size=size, api_type=api_type)


class ModelsConfig:
    def __init__(self):
        self._all_models = self._load_data()

    @property
    def all_models_configs(self) -> tuple[SingleModelConfig, ...]:
        return self._all_models

    @property
    def all_models(self) -> list[tuple[str, APIType | None]]:
        return list(self._all_models)

    @property
    def open_models(self) -> list[str]:
        return list(self._all_models)

    # TODO: filter/order by size, family, etc.

    @staticmethod
    def _load_data() -> tuple[SingleModelConfig, ...]:
        data_bytes = (_RESOURCES_PATH / "original-models-config.json").read_bytes()
        data_dict = json.loads(data_bytes)
        return tuple(SingleModelConfig.from_json_dict(s) for s in data_dict["models"])
    
    def __getitem__(self, item: str) -> SingleModelConfig:
        matches = [m for m in self._all_models if m.name == item]
        if not matches:
            raise KeyError(f"No model with name '{item}' found")
        if len(matches) > 1:
            logger.warning(f"Multiple models with name '{item}' found")
        return matches[0]
