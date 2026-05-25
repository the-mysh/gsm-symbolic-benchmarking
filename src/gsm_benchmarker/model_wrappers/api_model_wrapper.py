"""API-based model wrapper for remote inference.

Provides APIModelWrapper for unified access to cloud-hosted models via
OpenAI, Anthropic, and Google Gemini APIs. Handles provider-specific
client setup and inference interfaces.
"""

import logging
from typing import Callable

from gsm_benchmarker.model_wrappers.models_config_parser import SingleModelConfig, APIType


logger = logging.getLogger(__name__)

try:
    import openai
except ImportError:
    logger.warning("openai package not installed; OpenAI models will not be available")
    openai = None

try:
    import anthropic
except ImportError:
    logger.warning("anthropic package not installed; Anthropic models will not be available")
    anthropic = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    logger.warning("google-genai package not installed; Gemini family models will not be available")
    anthropic = None

from gsm_benchmarker.benchmark.benchmark_config import BenchmarkConfig
from gsm_benchmarker.model_wrappers.base_model_wrapper import BaseModelWrapper


class APIModelWrapper(BaseModelWrapper):
    """Wrapper for API-based models (OpenAI, Anthropic, Google Gemini).

    Provides a unified inference interface for cloud-hosted models by
    abstracting away provider-specific API details. Routes to appropriate
    client setup based on model configuration.
    """
    def __init__(self, model_spec: str | SingleModelConfig, config: BenchmarkConfig, api_type: APIType | None = None):
        """Initialize the API model wrapper.

        Args:
            model_spec: Model identifier or SingleModelConfig. If string,
                api_type must be explicitly provided.
            config: Benchmark configuration (temperature, max_tokens, etc.).
            api_type: Optional override for the API type. Uses model_spec's
                api_type if not provided.

        Raises:
            RuntimeError: If model_spec is a string and api_type is None.
        """
        if isinstance(model_spec, str) and api_type is None:
            raise RuntimeError(f"api_type must be specified for {self.__class__.__name__} either through model spec "
                               f"(given as a SingleModelConfig object) or through api_type arg; got neither")

        super().__init__(model_spec, config)

        if api_type is not None:
            self._model_spec.api_type = api_type

        self._client_interface = self._setup_client(self.config)

    def _setup_client(self, config: BenchmarkConfig) -> Callable[[str], str]:
        """Set up the appropriate API client based on model config.

        Args:
            config: Benchmark configuration.

        Returns:
            A callable that takes a prompt and returns the API response.

        Raises:
            TypeError: If api_type is not an APIType enum member.
            ValueError: If api_type is not recognized.
        """
        api_type = self._model_spec.api_type
        if not isinstance(api_type, APIType):
            raise TypeError(f"Expected an APIType enum member; got {type(api_type)}: {api_type}")

        match api_type:
            case APIType.openai:
                return self._setup_openai(self._model_spec, config)

            case APIType.anthropic:
                return self._setup_anthropic(self._model_spec, config)
            
            case APIType.google_genai:
                return self._setup_google_genai(self._model_spec, config)

            case _:
                raise ValueError(f"API type not recognised: {api_type}; expected {' / '.join(APIType.__members__)}")

    @staticmethod
    def _setup_openai(model_spec: SingleModelConfig, config: BenchmarkConfig) -> Callable[[str], str]:
        """Set up OpenAI API client.

        Args:
            model_spec: Model configuration with api model name.
            config: Benchmark configuration.

        Returns:
            A callable that queries the OpenAI API.

        Raises:
            ImportError: If openai package is not installed.
        """
        if openai is None:
            raise ImportError("openai package not installed")

        client = openai.OpenAI()

        def ask_openai(prompt: str) -> str:
            response = client.chat.completions.create(
                model=model_spec.name,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.temperature,
                max_tokens=config.max_new_tokens
            )
            return response.choices[0].message.content
        return ask_openai

    @staticmethod
    def _setup_anthropic(model_spec: SingleModelConfig, config: BenchmarkConfig) -> Callable[[str], str]:
        """Set up Anthropic (Claude) API client.

        Args:
            model_spec: Model configuration with Claude model name.
            config: Benchmark configuration.

        Returns:
            A callable that queries the Anthropic API.

        Raises:
            ImportError: If anthropic package is not installed.
        """
        if anthropic is None:
            raise ImportError("anthropic package not installed")

        client = anthropic.Anthropic()
        def ask_anthropic(prompt: str) -> str:
            message = client.messages.create(
                model=model_spec.name,
                max_tokens=config.max_new_tokens,
                temperature=config.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        return ask_anthropic

    @staticmethod
    def _setup_google_genai(model_spec: SingleModelConfig, config: BenchmarkConfig) -> Callable[[str], str]:
        """Set up Google Gemini API client.

        Args:
            model_spec: Model configuration with Gemini model name.
            config: Benchmark configuration.

        Returns:
            A callable that queries the Google Gemini API.

        Raises:
            ImportError: If google-genai package is not installed.
        """
        if genai is None:
            raise ImportError("google-genai package not installed")

        client = genai.Client()

        generation_config = types.GenerateContentConfig(
            temperature=config.temperature,
            max_output_tokens=config.max_new_tokens
        )

        def ask_google_genai(prompt: str) -> str:
            response = client.models.generate_content(
                model=model_spec.name,
                contents=prompt,
                config=generation_config
            )
            return response.text

        return ask_google_genai


    def ask(self, prompt: str) -> str:
        """Generate a response using the API.

        Args:
            prompt: The input prompt.

        Returns:
            The API model's response.
        """
        return self._client_interface(prompt)
