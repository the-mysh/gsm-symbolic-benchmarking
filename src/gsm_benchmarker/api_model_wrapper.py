import logging
from typing import Callable
from enum import Enum, auto

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

from gsm_benchmarker.benchmark_config import BenchmarkConfig
from gsm_benchmarker.base_model_wrapper import BaseModelWrapper


class APIType(Enum):
    openai = auto()
    anthropic = auto()
    google_genai = auto()


class APIModelWrapper(BaseModelWrapper):
    def __init__(self, model_name: str, config: BenchmarkConfig, api_type: APIType):
        super().__init__(model_name, config)

        self._client_interface = self._setup_client(self._model_name, api_type, self.config)

    def _setup_client(self, model_name: str, api_type: APIType, config: BenchmarkConfig) -> Callable[[str], str]:
        if not isinstance(api_type, APIType):
            raise TypeError(f"Expected an APIType enum member; got {type(api_type)}: {api_type}")

        match api_type:
            case APIType.openai:
                return self._setup_openai(model_name, config)

            case APIType.anthropic:
                return self._setup_anthropic(model_name, config)
            
            case APIType.google_genai:
                return self._setup_google_genai(model_name, config)

            case _:
                raise ValueError(f"API type not recognised: {api_type}; expected {' / '.join(APIType.__members__)}")

    @staticmethod
    def _setup_openai(model_name: str, config: BenchmarkConfig) -> Callable[[str], str]:
        if openai is None:
            raise ImportError("openai package not installed")

        def ask_openai(prompt: str) -> str:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.temperature,
                max_tokens=config.max_new_tokens
            )
            return response.choices[0].message.content
        return ask_openai

    @staticmethod
    def _setup_anthropic(model_name: str, config: BenchmarkConfig) -> Callable[[str], str]:
        if anthropic is None:
            raise ImportError("anthropic package not installed")

        client = anthropic.Anthropic()
        def ask_anthropic(prompt: str) -> str:
            message = client.messages.create(
                model=model_name,
                max_tokens=config.max_new_tokens,
                temperature=config.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        return ask_anthropic

    @staticmethod
    def _setup_google_genai(model_name: str, config: BenchmarkConfig) -> Callable[[str], str]:
        if genai is None:
            raise ImportError("google-genai package not installed")

        client = genai.Client()

        generation_config = types.GenerateContentConfig(
            temperature=config.temperature,
            max_output_tokens=config.max_new_tokens
        )

        def ask_google_genai(prompt: str) -> str:
            message = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=generation_config
            )
            return message.content[0].text

        return ask_google_genai


    def ask(self, prompt: str) -> str:
        return self._client_interface(prompt)
