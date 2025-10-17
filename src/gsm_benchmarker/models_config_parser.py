import json
from dataclasses import dataclass
from importlib.resources import files


from gsm_benchmarker.api_model_wrapper import APIType


_RESOURCES_PATH = files("gsm_benchmarker")/"resources"



@dataclass
class SingleModelConfig:
    name: str
    family: str
    size: float | None
    instruction_tuned: bool
    api_type: APIType | None

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
        return [(m.name, m.api_type) for m in self._all_models]

    @property
    def open_models(self) -> list[str]:
        return [m.name for m in self._all_models if m.api_type is None]

    # TODO: filter/order by size, family, etc.

    @staticmethod
    def _load_data() -> tuple[SingleModelConfig, ...]:
        data_bytes = (_RESOURCES_PATH / "original-models-config.json").read_bytes()
        data_dict = json.loads(data_bytes)
        return tuple(SingleModelConfig.from_json_dict(s) for s in data_dict["models"])