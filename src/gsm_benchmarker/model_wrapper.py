import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

from gsm_benchmarker.benchmark_config import BenchmarkConfig


logger = logging.getLogger(__name__)


class HFModelWrapper:
    def __init__(self, model_name: str, config: BenchmarkConfig):
        self.config = config

        logger.info(f"Setting up model {model_name}")

        self.tokeniser = self.load_tokeniser(model_name, trust_remote_code=self.config.trust_remote_code)
        self.model = self.load_model(model_name)
        logger.info("Model loaded")

    @staticmethod
    def load_tokeniser(model_name: str, trust_remote_code: bool = False):
        logger.debug("Loading tokeniser")
        tokeniser = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code
        )

        # Set padding token if not set
        if tokeniser.pad_token is None:
            tokeniser.pad_token = tokeniser.eos_token

        return tokeniser

    @staticmethod
    def load_model(model_name):
        if torch.cuda.is_available():
            logger.info("CUDA available")
            model = HFModelWrapper.load_model_cuda(model_name)
        else:
            logger.info("CUDA not available - using only CPU")
            model = HFModelWrapper.load_model_cpu(model_name)

        return model

    @staticmethod
    def load_model_cuda(model_name):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,  # Extra compression
            bnb_4bit_quant_type="nf4"
        )

        logger.debug("Loading model with CUDA")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            max_memory={0: "7GB", "cpu": "12GB"}  # Adjust based on your hardware
        )

        return model

    @staticmethod
    def load_model_cpu(model_name):
        logger.debug("Loading model for CPU only")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            low_cpu_mem_usage=True
        )

        return model

    def ask(self, prompt: str) -> str:
        """Generate response from model"""
        inputs = self.tokeniser(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature if self.config.temperature > 0 else 1.0,
                do_sample=self.config.temperature > 0,
                pad_token_id=self.tokeniser.pad_token_id,
                eos_token_id=self.tokeniser.eos_token_id,
            )

        # Decode only the generated part (not the prompt)
        generated_text = self.tokeniser.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return generated_text
