"""Base class defining the model wrapper interface.

Provides the abstract BaseModelWrapper that defines the contract for all
model wrappers (HuggingFace and API-based).
"""

from gsm_benchmarker.benchmark.benchmark_config import BenchmarkConfig
from gsm_benchmarker.model_wrappers.models_config_parser import SingleModelConfig
import logging


logger = logging.getLogger(__name__)


class BaseModelWrapper:
    """Abstract base class for model wrappers.

    Defines the unified interface for all model types, handling configuration
    initialization and model identification. Concrete subclasses implement
    the `ask()` method with provider-specific inference logic.
    """
    def __init__(self, model_spec: str | SingleModelConfig, config: BenchmarkConfig):
        """Initialize the base wrapper.

        Args:
            model_spec: Either a model name string or a SingleModelConfig object.
                If a string, a default SingleModelConfig is created.
            config: Benchmark configuration (inference params, memory settings, etc.).

        Raises:
            TypeError: If model_spec is neither a string nor SingleModelConfig.
        """
        if isinstance(model_spec, str):
            logger.debug(f"Constructing default model spec object for model {model_spec}")
            model_spec = SingleModelConfig(model_spec)
        if not isinstance(model_spec, SingleModelConfig):
            raise TypeError(f"Expected a SingleModelConfig object or a str; got {type(model_spec)}: {model_spec}")

        self.config = config
        self._model_spec = model_spec

    @property
    def model_name(self) -> str:
        """Return the model name from the spec."""
        return self._model_spec.name

    def ask(self, prompt: str) -> str:
        """Generate a response from the model.

        Args:
            prompt: The input prompt for the model.

        Returns:
            The model's text response.

        Raises:
            NotImplementedError: This is an abstract method; subclasses must implement.
        """

        raise NotImplementedError
