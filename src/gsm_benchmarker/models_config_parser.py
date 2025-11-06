from dataclasses import dataclass
from typing import Any
import logging
from enum import Enum, auto

from gsm_benchmarker.utils.resources_manager import load_resource_json


logger = logging.getLogger(__name__)


class APIType(Enum):
    openai = auto()
    anthropic = auto()
    google_genai = auto()


@dataclass
class SingleModelConfig:
    name: str
    family: str = ""
    size: float | None = None
    instruction_tuned: bool | None = False
    api_type: APIType | None = None
    extra_kwargs: dict[str, dict[str, Any]] = None
    trust_remote_code: bool = False

    def __post_init__(self):
        if self.extra_kwargs is None:
            self.extra_kwargs = {}

    @property
    def extra_kwargs_model_init(self):
        return self.extra_kwargs.get("from_pretrained", {})

    @property
    def extra_kwargs_tokeniser_init(self):
        return self.extra_kwargs.get("tokeniser_from_pretrained", {})

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
    def all_models(self) -> list[SingleModelConfig]:
        return list(self._all_models)

    @property
    def open_models(self) -> list[SingleModelConfig]:
        return [m for m in self._all_models if m.api_type is None]

    # TODO: filter/order by size, family, etc.

    @staticmethod
    def _load_data() -> tuple[SingleModelConfig, ...]:
        data_dict = load_resource_json("original-models-config.json")
        return tuple(SingleModelConfig.from_json_dict(s) for s in data_dict["models"])
    
    def __getitem__(self, item: str) -> SingleModelConfig:
        matches = [m for m in self._all_models if m.name == item]
        if not matches:
            raise KeyError(f"No model with name '{item}' found")
        if len(matches) > 1:
            logger.warning(f"Multiple models with name '{item}' found")
        return matches[0]
