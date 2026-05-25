"""Model wrapper abstraction layer for unified inference interface.

This subpackage provides a unified interface for different model types:

* `BaseModelWrapper`: Abstract interface defining the model wrapper contract.
* `HFModelWrapper`: Concrete implementation for HuggingFace transformer models
  (local inference with CUDA/CPU support and quantization).
* `APIModelWrapper`: Concrete implementation for remote API-based models
  (OpenAI, Anthropic, Google Gemini).
* `ModelsConfig` & `SingleModelConfig`: Configuration management and model
  metadata parsing from JSON resources.

**Factory pattern:** ModelEvaluator automatically dispatches to the appropriate
wrapper based on model configuration (api_type).
"""
from gsm_benchmarker.model_wrappers.base_model_wrapper import BaseModelWrapper
from gsm_benchmarker.model_wrappers.hf_model_wrapper import HFModelWrapper
from gsm_benchmarker.model_wrappers.api_model_wrapper import APIModelWrapper
