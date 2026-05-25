"""Benchmark configuration management.

Provides centralized configuration for model inference, memory allocation, and
GPU/CPU setup. Supports both direct configuration and machine-specific presets
loaded from JSON resources.
"""

from dataclasses import dataclass, asdict
import torch
from typing import Any
import logging

from gsm_benchmarker.utils.resources_manager import load_resource_json


_AUTO = object()
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for GSM-Symbolic benchmark execution.

    This dataclass centralizes all settings needed to run a benchmark: model
    inference parameters (temperature, generation length), memory allocation
    (CPU/GPU limits), and compute resource selection. It supports both direct
    instantiation with explicit parameters and a factory method to load
    machine-specific presets.

    Attributes:
        temperature: Sampling temperature for generation (0.0 = greedy).
        max_new_tokens: Maximum tokens to generate per example.
        max_length: Maximum sequence length (context + generation).
        use_4bit: Whether to use 4-bit quantization for memory efficiency.
        trust_remote_code_global: Allow loading of remote code from HuggingFace Hub.
        native_dtype: Torch dtype for model inference (float16 or bfloat16).
        gpu_max_memory: Maximum VRAM to allocate per GPU (GiB), or None for CPU-only.
        gpu_index: Which GPU to use (0-indexed), _AUTO for auto-detect, or None for CPU-only.
        gpu_auto: If True, enable automatic GPU memory management.
        cpu_max_memory: Maximum CPU RAM to allocate (GiB).
    """

    temperature: float = 0.0  # greedy decoding
    max_new_tokens: int = 1024
    max_length: int = 2048
    use_4bit: bool = True  # for memory efficiency
    trust_remote_code_global: bool = False
    native_dtype: type[torch.float] = torch.float16

    # memory settings
    gpu_max_memory: int | None = 7
    gpu_index: int | None | type[_AUTO] = _AUTO
    gpu_auto: bool = False
    cpu_max_memory: int = 12

    def __post_init__(self):
        """Resolve automatic GPU index if not explicitly set.

        If gpu_index is _AUTO and CUDA is available, sets it to 0;
        otherwise sets it to None (CPU-only).
        """
        if self.gpu_index is _AUTO:
            if torch.cuda.is_available():
                logger.info("Setting default gpu index: 0")
                self.gpu_index = 0
            else:
                logger.info("No GPUs available")
                self.gpu_index = None

    def to_dict(self):
        """Convert configuration to dictionary."""
        return asdict(self)

    @property
    def memory_settings(self) -> dict[str | int, str]:
        """Get memory allocation as a dictionary for PyTorch accelerate.

        Returns:
            Dictionary mapping 'cpu' to CPU memory limit (e.g. "12GiB") and
            GPU index (if gpu_index is set) to VRAM limit (e.g. "8GiB").

        Raises:
            RuntimeError: If gpu_index is set but gpu_max_memory is not defined.
        """
        mem: dict[str | int, str] = {"cpu": f"{self.cpu_max_memory}GiB"}

        if self.gpu_index is not None:
            if not self.gpu_max_memory:
                raise RuntimeError("gpu_max_memory is not defined")
            mem[self.gpu_index] = f"{self.gpu_max_memory}GiB"
        
        return mem

    @classmethod
    def for_machine(cls, machine_name: str, gpu_index: int | None | type[_AUTO] = _AUTO, ram_margin=4, vram_margin=2,
                    **kwargs: Any) -> "BenchmarkConfig":
        """Create a configuration for a specific machine by name.

        Loads machine presets from resources/machines_config.json and calculates
        optimal memory allocation based on machine type and specified margins.

        Args:
            machine_name: Name of the machine (must exist in machines_config.json).
            gpu_index: GPU index to use, _AUTO to auto-detect, or None for CPU-only.
            ram_margin: RAM margin (GiB) to reserve from total available.
            vram_margin: VRAM margin (GiB) to reserve from total available.
            **kwargs: Additional BenchmarkConfig parameters to override.

        Returns:
            A BenchmarkConfig instance configured for the specified machine.

        Raises:
            ValueError: If machine_name is not defined or GPU index is invalid.
        """

        machines_config = load_resource_json('machines_config.json')

        machine_type: str | None = machines_config['machines'].get(machine_name, None)
        if not machine_type:
            raise ValueError(f"No configuration defined for machine name '{machine_name}'")

        logger.debug(f"Reading config for machine '{machine_name}', of type '{machine_type}'")

        machine_params: dict | None = machines_config['machine_types'].get(machine_type, None)
        if not machine_params:
            raise RuntimeError(f"No configuration defined for machine type '{machine_type}'")

        n_gpus = machine_params.get('gpus')
        assert isinstance(n_gpus, int)

        native_dtype = torch.bfloat16 if machine_params['bfloat'] else torch.float16

        cls.validate_gpu_index(machine_name, n_gpus, gpu_index)
        cpu_max_memory, gpu_max_memory = cls.get_max_memories(machine_params, gpu_index, ram_margin, vram_margin)

        if 'gpu_max_memory' not in kwargs:
            kwargs['gpu_max_memory'] = gpu_max_memory

        if 'cpu_max_memory' not in kwargs:
            kwargs['cpu_max_memory'] = cpu_max_memory

        return BenchmarkConfig(
            gpu_index=gpu_index,
            native_dtype=native_dtype,
            **kwargs
        )

    @staticmethod
    def validate_gpu_index(machine_name: str, n_gpus: int, gpu_index: int | None | type[_AUTO] = _AUTO) -> None:
        """Validate that the requested GPU index is valid for the machine.

        Args:
            machine_name: Name of the machine (used in error messages).
            n_gpus: Total number of GPUs on the machine.
            gpu_index: GPU index to validate, _AUTO to skip, or None for CPU-only.

        Raises:
            ValueError: If gpu_index is an integer and >= n_gpus.
        """
        if gpu_index is _AUTO:
            return  # auto-defined in __post_init__

        if gpu_index is None and n_gpus:
            logger.warning(f"gpu_index is set to None; none of the available {n_gpus} GPUs will be used")

        if gpu_index is not None and gpu_index >= n_gpus:
            raise ValueError(f"Cannot use GPU {gpu_index} for machine '{machine_name}' with a total of {n_gpus} GPUs")

    @staticmethod
    def get_max_memories(machine_params: dict, gpu_index: int | None, ram_margin: int, vram_margin: int
                         ) -> tuple[int, int | None]:
        """Calculate maximum usable memory after subtracting safety margins.

        Args:
            machine_params: Machine parameters dict with 'ram' and 'vram' keys (in GiB).
            gpu_index: GPU index; if None, gpu_memory is not calculated.
            ram_margin: Amount of RAM (GiB) to reserve.
            vram_margin: Amount of VRAM (GiB) to reserve.

        Returns:
            Tuple of (cpu_memory_gib, gpu_memory_gib) with margins subtracted.
            gpu_memory_gib is None if gpu_index is None.
        """
        cpu_memory = machine_params.get('ram')
        assert isinstance(cpu_memory, int)
        cpu_memory -= ram_margin

        if gpu_index is None:
            gpu_memory = None
        else:
            gpu_memory = machine_params.get('vram')
            assert isinstance(gpu_memory, int)
            gpu_memory -= vram_margin

        return cpu_memory, gpu_memory
