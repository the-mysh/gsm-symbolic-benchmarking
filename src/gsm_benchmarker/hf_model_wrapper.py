import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import scan_cache_dir
import torch

from gsm_benchmarker.benchmark_config import BenchmarkConfig
from gsm_benchmarker.base_model_wrapper import BaseModelWrapper
from gsm_benchmarker.models_config_parser import SingleModelConfig

logger = logging.getLogger(__name__)


class HFModelWrapper(BaseModelWrapper):
    def __init__(self, model_spec: str | SingleModelConfig, config: BenchmarkConfig):
        super().__init__(model_spec, config)

        logger.info(f"Setting up model {self.model_name}")
        self.tokeniser = self._load_tokeniser(config=self.config)
        self.model = self._load_model(config=self.config)
        logger.info("Model loaded")

    def _load_tokeniser(self, config: BenchmarkConfig):
        logger.debug("Loading tokeniser")
        
        extras = self._model_spec.extra_kwargs_tokeniser_init
        if extras:
            logger.debug(f"Passing extra kwargs to 'AutoTokenizer.from_pretrained': {extras}")
            
        tokeniser = AutoTokenizer.from_pretrained(
            self._model_spec.name,
            trust_remote_code=config.trust_remote_code_global and self._model_spec.trust_remote_code,
            **extras
        )

        # Set padding token if not set
        if tokeniser.pad_token is None:
            tokeniser.pad_token = tokeniser.eos_token
            tokeniser.pad_token_id = tokeniser.eos_token_id  # for generation

        tokeniser.padding_side = "left"

        return tokeniser

    def _load_model(self, config: BenchmarkConfig):
        if torch.cuda.is_available():
            logger.debug("CUDA available")
            model = self._load_model_cuda(config=config)
        else:
            logger.debug("CUDA not available - using only CPU")
            model = self._load_model_cpu(config=config)

        return model

    def _load_model_cuda(self, config: BenchmarkConfig):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config.use_4bit,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,  # Extra compression
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True
        )

        logger.debug("Loading model with CUDA")
        extras = self._model_spec.extra_kwargs_model_init
        if extras:
            logger.debug(f"Passing extra kwargs to 'AutoModelForCausalLM.from_pretrained': {extras}")

        model = AutoModelForCausalLM.from_pretrained(
            self._model_spec.name,
            quantization_config=bnb_config,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            max_memory=config.memory_settings,
            trust_remote_code=config.trust_remote_code_global and self._model_spec.trust_remote_code,
            **extras
        )

        return model

    def _load_model_cpu(self, config: BenchmarkConfig):
        logger.debug("Loading model for CPU only")
        
        extras = self._model_spec.extra_kwargs_model_init
        if extras:
            logger.debug(f"Passing extra kwargs to 'AutoModelForCausalLM.from_pretrained': {extras}")
        
        model = AutoModelForCausalLM.from_pretrained(
            self._model_spec.name,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=config.trust_remote_code_global,
            **extras
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
    
    def delete_from_cache(self):
        del self.model
        del self.tokeniser

        # Scan the entire cache directory
        cache_info = scan_cache_dir()
        
        # Find the CachedRepoInfo object for the specific model
        repo_to_delete = None
        for repo_info in cache_info.repos:
            if repo_info.repo_id == self.model_name:
                repo_to_delete = repo_info
                break
                
        if repo_to_delete is None:
            logger.debug(f"Model '{self.model_name}' not found in cache. Skipping deletion.")
            return

        # Get all revision hashes associated with that model
        # Create the deletion strategy based on the model's revision hashes
        # This automatically finds all files (blobs) unique to this model's revisions
        revisions_to_delete = {rev.commit_hash for rev in repo_to_delete.revisions}
        delete_strategy = cache_info.delete_revisions(*revisions_to_delete)
        
        # perform the actual file deletion
        logger.debug("Deleting model {model_repo_id}")
        delete_strategy.execute()
        
        logger.debug(f"Model {self.model_name} deleted from cache")


