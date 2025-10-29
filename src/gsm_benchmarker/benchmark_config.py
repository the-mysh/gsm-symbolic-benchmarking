from dataclasses import dataclass
import torch


@dataclass
class BenchmarkConfig:
    """Configuration for GSM-Symbolic benchmark"""
    random_seed: int = 42
    num_shots: int = 8
    temperature: float = 0.0  # greedy decoding
    max_new_tokens: int = 1024
    batch_size: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Model-specific
    use_4bit: bool = True  # For memory efficiency
    trust_remote_code_global: bool = False

    # memory settings
    gpu0_max_memory: str | None = "7GB"
    gpu1_max_memory: str | None = None
    cpu_max_memory: str = "12GB"

    @property
    def memory_settings(self):
        mem = {"cpu": self.cpu_max_memory}
        if self.gpu0_max_memory is not None:
            mem[0] = self.gpu0_max_memory
        
        if self.gpu1_max_memory is not None:
                mem[1] = self.gpu1_max_memory
        
        return mem
    
