import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import scan_cache_dir
import torch

from gsm_benchmarker.benchmark_config import BenchmarkConfig
from gsm_benchmarker.base_model_wrapper import BaseModelWrapper


logger = logging.getLogger(__name__)


class HFModelWrapper(BaseModelWrapper):
    def __init__(self, model_name: str, config: BenchmarkConfig):
        super().__init__(model_name, config)

        logger.info(f"Setting up model {model_name}")
        self._model_name = model_name
        self.tokeniser = self._load_tokeniser(model_name, trust_remote_code=self.config.trust_remote_code)
        self.model = self._load_model(model_name, config=self.config)
        logger.info("Model loaded")

    @staticmethod
    def _load_tokeniser(model_name: str, trust_remote_code: bool = False):
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
    def _load_model(model_name, config: BenchmarkConfig):
        if torch.cuda.is_available():
            logger.info("CUDA available")
            model = HFModelWrapper._load_model_cuda(model_name, config=config)
        else:
            logger.info("CUDA not available - using only CPU")
            model = HFModelWrapper._load_model_cpu(model_name, config=config)

        return model

    @staticmethod
    def _load_model_cuda(model_name, config: BenchmarkConfig):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,  # Extra compression
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True
        )

        logger.debug("Loading model with CUDA")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            max_memory={0: config.gpu0_max_memory, "cpu": config.cpu_max_memory},
            trust_remote_code=config.trust_remote_code
        )

        return model

    @staticmethod
    def _load_model_cpu(model_name, config: BenchmarkConfig):
        logger.debug("Loading model for CPU only")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=config.trust_remote_code
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
            if repo_info.repo_id == self._model_name:
                repo_to_delete = repo_info
                break
                
        if repo_to_delete is None:
            logger.info(f"Model '{self._model_name}' not found in cache. Skipping deletion.")
            return

        # Get all revision hashes associated with that model
        # Create the deletion strategy based on the model's revision hashes
        # This automatically finds all files (blobs) unique to this model's revisions
        revisions_to_delete = {rev.commit_hash for rev in repo_to_delete.revisions}
        delete_strategy = cache_info.delete_revisions(*revisions_to_delete)
        
        # perform the actual file deletion
        logger.debug("Deleting model {model_repo_id}")
        delete_strategy.execute()
        
        logger.info(f"Model {self._model_name} deleted from cache")


