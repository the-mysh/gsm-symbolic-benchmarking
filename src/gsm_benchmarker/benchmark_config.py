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
    trust_remote_code: bool = False
