from dataclasses import dataclass, asdict
import torch
from typing import Any
import logging

from gsm_benchmarker.utils.resources_manager import load_resource_json


_AUTO = object()
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for GSM-Symbolic benchmark"""

    temperature: float = 0.0  # greedy decoding
    max_new_tokens: int = 1024
    use_4bit: bool = True  # for memory efficiency
    trust_remote_code_global: bool = False

    # memory settings
    gpu_max_memory: int | None = 7
    gpu_index: int | None | type[_AUTO] = _AUTO
    cpu_max_memory: int = 12

    def __post_init__(self):
        if self.gpu_index is _AUTO:
            if torch.cuda.is_available():
                logger.info("Setting default gpu index: 0")
                self.gpu_index = 0
            else:
                logger.info("No GPUs available")
                self.gpu_index = None

    def to_dict(self):
        return asdict(self)

    @property
    def memory_settings(self):
        mem: dict[str | int, str] = {"cpu": f"{self.cpu_max_memory}GiB"}

        if self.gpu_index is not None:
            if not self.gpu_max_memory:
                raise RuntimeError("gpu_max_memory is not defined")
            mem[self.gpu_index] = f"{self.gpu_max_memory}GiB"
        
        return mem

    @classmethod
    def for_machine(cls, machine_name: str, gpu_index: int | None | type[_AUTO] = _AUTO, ram_margin=4, vram_margin=2,
                    **kwargs: Any) -> "BenchmarkConfig":

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

        cls.validate_gpu_index(machine_name, n_gpus, gpu_index)
        cpu_max_memory, gpu_max_memory = cls.get_max_memories(machine_params, gpu_index, ram_margin, vram_margin)

        if 'gpu_max_memory' not in kwargs:
            kwargs['gpu_max_memory'] = gpu_max_memory

        if 'cpu_max_memory' not in kwargs:
            kwargs['cpu_max_memory'] = cpu_max_memory

        return BenchmarkConfig(
            gpu_index=gpu_index,
            **kwargs
        )

    @staticmethod
    def validate_gpu_index(machine_name: str, n_gpus: int, gpu_index: int | None | type[_AUTO] = _AUTO) -> None:

        if gpu_index is _AUTO:
            return  # auto-defined in __post_init__

        if gpu_index is None and n_gpus:
            logger.warning(f"gpu_index is set to None; none of the available {n_gpus} GPUs will be used")

        if gpu_index is not None and gpu_index >= n_gpus:
            raise ValueError(f"Cannot use GPU {gpu_index} for machine '{machine_name}' with a total of {n_gpus} GPUs")

    @staticmethod
    def get_max_memories(machine_params: dict, gpu_index: int | None, ram_margin: int, vram_margin: int
                         ) -> tuple[int, int | None]:
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
