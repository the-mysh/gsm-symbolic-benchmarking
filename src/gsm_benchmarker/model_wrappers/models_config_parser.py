"""Model configuration parsing and metadata management.

Provides configuration classes for model specifications and metadata loading
from JSON resources, supporting both local HuggingFace models and remote
API-based models.
"""

from dataclasses import dataclass
from typing import Any
import logging
from enum import Enum, auto

from gsm_benchmarker.utils.resources_manager import load_resource_json


logger = logging.getLogger(__name__)


class APIType(Enum):
    """Enumeration of supported API-based model providers.

    Attributes:
        openai: OpenAI models (gpt-3.5-turbo, gpt-4, etc.).
        anthropic: Anthropic models (Claude family).
        google_genai: Google Gemini models.
    """
    openai = auto()
    anthropic = auto()
    google_genai = auto()


@dataclass
class SingleModelConfig:
    """Configuration for a single model.

    Stores model metadata, resource specifications, and provider information.
    Supports both local models (HuggingFace) and API-based models with
    provider-specific extra kwargs.

    Attributes:
        name: Model identifier (HF model ID or API model name).
        family: Model family/series (e.g., "llama", "gpt", "claude").
        size: Model parameter count in billions, or None if unknown.
        instruction_tuned: Whether the model is instruction-fine-tuned.
        api_type: APIType if remote API model, None for local HuggingFace models.
        extra_kwargs: Dict mapping kwarg categories to dicts of provider-specific args.
            - "from_pretrained": Extra kwargs for AutoModelForCausalLM.from_pretrained
            - "tokeniser_from_pretrained": Extra kwargs for AutoTokenizer.from_pretrained
        trust_remote_code: Whether to allow remote code execution from HF Hub.
    """
    name: str
    family: str = ""
    size: float | None = None
    instruction_tuned: bool | None = False
    api_type: APIType | None = None
    extra_kwargs: dict[str, dict[str, Any]] = None
    trust_remote_code: bool = False

    def __post_init__(self):
        """Initialize extra_kwargs to empty dict if None."""
        if self.extra_kwargs is None:
            self.extra_kwargs = {}

    @property
    def extra_kwargs_model_init(self):
        """Extra kwargs for AutoModelForCausalLM.from_pretrained()."""
        return self.extra_kwargs.get("from_pretrained", {})

    @property
    def extra_kwargs_tokeniser_init(self):
        """Extra kwargs for AutoTokenizer.from_pretrained()."""
        return self.extra_kwargs.get("tokeniser_from_pretrained", {})

    @classmethod
    def from_json_dict(cls, d: dict):
        """Create a config from a dictionary (e.g., from JSON).

        Parses api_type string to enum and size to float.

        Args:
            d: Dictionary with config fields.

        Returns:
            A SingleModelConfig instance.
        """
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
    """Manager for all available models and their configurations.

    Loads model metadata from resources/original-models-config.json and
    provides filtering and lookup by name.
    """
    def __init__(self):
        """Load all model configurations from resources."""
        self._all_models = self._load_data()

    @property
    def all_models_configs(self) -> tuple[SingleModelConfig, ...]:
        """Return all model configs as a tuple."""
        return self._all_models

    @property
    def all_models(self) -> list[SingleModelConfig]:
        """Return all model configs as a list."""
        return list(self._all_models)

    @property
    def open_models(self) -> list[SingleModelConfig]:
        """Return only models that are open/local (no API required).

        Returns models where api_type is None.
        """
        return [m for m in self._all_models if m.api_type is None]

    @staticmethod
    def _load_data() -> tuple[SingleModelConfig, ...]:
        """Load model data from resources JSON file.

        Returns:
            Tuple of SingleModelConfig objects.
        """
        data_dict = load_resource_json("original-models-config.json")
        return tuple(SingleModelConfig.from_json_dict(s) for s in data_dict["models"])
    
    def __getitem__(self, item: str) -> SingleModelConfig:
        """Look up a model config by name.

        Args:
            item: Model name to find.

        Returns:
            The matching SingleModelConfig.

        Raises:
            KeyError: If no model with the given name exists.
        """
        matches = [m for m in self._all_models if m.name == item]
        if not matches:
            raise KeyError(f"No model with name '{item}' found")
        if len(matches) > 1:
            logger.warning(f"Multiple models with name '{item}' found")
        return matches[0]
