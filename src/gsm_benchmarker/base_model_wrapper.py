from gsm_benchmarker.benchmark_config import BenchmarkConfig
from gsm_benchmarker.models_config_parser import SingleModelConfig
import logging


logger = logging.getLogger(__name__)


class BaseModelWrapper:
    def __init__(self, model_spec: str | SingleModelConfig, config: BenchmarkConfig):
        if isinstance(model_spec, str):
            logger.debug(f"Constructing default model spec object for model {model_spec}")
            model_spec = SingleModelConfig(model_spec)
        if not isinstance(model_spec, SingleModelConfig):
            raise TypeError(f"Expected a SingleModelConfig object or a str; got {type(model_spec)}: {model_spec}")

        self.config = config
        self._model_spec = model_spec

    @property
    def model_name(self) -> str:
        return self._model_spec.name

    def ask(self, prompt: str) -> str:
        """Generate response from model"""

        raise NotImplementedError
