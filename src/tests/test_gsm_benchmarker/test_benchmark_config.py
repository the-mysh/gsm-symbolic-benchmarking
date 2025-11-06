# written with the help of Gemini

import pytest
import logging
from unittest.mock import patch

from gsm_benchmarker.benchmark_config import BenchmarkConfig, _AUTO
from gsm_benchmarker.utils.resources_manager import load_resource_json


# --- Fixtures for Mocking External Dependencies ---

@pytest.fixture
def machines_config():
    """Fixture providing the mocked content of machines_config.json."""
    return load_resource_json('machines_config.json')


@pytest.fixture(autouse=True)
def mock_load_resource_json(machines_config):
    """
    Globally patch the external function to return our fixture data.
    """
    with patch('gsm_benchmarker.benchmark_config.load_resource_json', return_value=machines_config) as mock:
        yield mock


# --- Tests for Initialization (__post_init__) ---

@patch('gsm_benchmarker.benchmark_config.torch.cuda.is_available', return_value=True)
def test_post_init_with_gpu_available(mock_cuda_avail, caplog):
    """Test that gpu_index is set to 0 when CUDA is available and index is _AUTO."""
    caplog.set_level(logging.INFO)
    config = BenchmarkConfig(gpu_index=_AUTO)
    assert config.gpu_index == 0
    assert "Setting default gpu index: 0" in caplog.text


@patch('gsm_benchmarker.benchmark_config.torch.cuda.is_available', return_value=False)
def test_post_init_with_no_gpu_available(mock_cuda_avail, caplog):
    """Test that gpu_index is set to None when CUDA is not available and index is _AUTO."""
    caplog.set_level(logging.INFO)
    config = BenchmarkConfig(gpu_index=_AUTO)
    assert config.gpu_index is None
    assert "No GPUs available" in caplog.text


def test_post_init_with_explicit_index():
    """Test that an explicitly set gpu_index is preserved."""
    config = BenchmarkConfig(gpu_index=2)
    assert config.gpu_index == 2


# --- Tests for memory_settings property ---

def test_memory_settings_with_gpu():
    """Test memory settings when a GPU is configured."""
    config = BenchmarkConfig(cpu_max_memory=12, gpu_index=1, gpu_max_memory=8)
    expected_mem = {"cpu": "12GiB", 1: "8GiB"}
    assert config.memory_settings == expected_mem


def test_memory_settings_without_gpu():
    """Test memory settings when no GPU is configured (gpu_index is None)."""
    config = BenchmarkConfig(cpu_max_memory=6, gpu_index=None)
    expected_mem = {"cpu": "6GiB"}
    assert config.memory_settings == expected_mem


def test_memory_settings_raises_on_missing_gpu_max_memory():
    """Test RuntimeError is raised if gpu_index is set but gpu_max_memory is not."""
    config = BenchmarkConfig(gpu_index=0, gpu_max_memory=None)
    with pytest.raises(RuntimeError, match="gpu_max_memory is not defined"):
        _ = config.memory_settings


# --- Tests for get_max_memories (Static Method) ---

def test_get_max_memories_with_gpu_index():
    """Test memory calculation when a GPU index is provided."""
    machine_params = {"ram": 100, "vram": 40, "gpus": 2}
    cpu_mem, gpu_mem = BenchmarkConfig.get_max_memories(machine_params, gpu_index=1, ram_margin=5, vram_margin=2)

    # Expected: 100 - 5 = 95GiB (CPU), 40 - 2 = 38GiB (GPU)
    assert cpu_mem == 95
    assert gpu_mem == 38


def test_get_max_memories_without_gpu_index():
    """Test memory calculation when gpu_index is None."""
    machine_params = {"ram": 100, "vram": 40, "gpus": 2}
    cpu_mem, gpu_mem = BenchmarkConfig.get_max_memories(machine_params, gpu_index=None, ram_margin=10, vram_margin=2)

    # Expected: 100 - 10 = 90GiB (CPU), None (GPU)
    assert cpu_mem == 90
    assert gpu_mem is None


# --- Tests for validate_gpu_index (Static Method) ---

def test_validate_gpu_index_auto():
    """Test no action for _AUTO index."""
    # Should not raise
    BenchmarkConfig.validate_gpu_index("test", 1, _AUTO)


def test_validate_gpu_index_none_with_gpus(caplog):
    """Test warning log when gpu_index is None but GPUs are available."""
    caplog.set_level(logging.WARNING)
    BenchmarkConfig.validate_gpu_index("test", 4, None)
    assert "gpu_index is set to None; none of the available 4 GPUs will be used" in caplog.text


def test_validate_gpu_index_valid_index():
    """Test valid GPU index passes silently."""
    # Should not raise
    BenchmarkConfig.validate_gpu_index("test", 4, 3)
    BenchmarkConfig.validate_gpu_index("test", 1, 0)


def test_validate_gpu_index_invalid_index():
    """Test invalid GPU index raises ValueError."""
    with pytest.raises(ValueError, match="Cannot use GPU 2 for machine 'test' with a total of 2 GPUs"):
        BenchmarkConfig.validate_gpu_index("test", 2, 2)

    with pytest.raises(ValueError, match="Cannot use GPU 5 for machine 'another' with a total of 4 GPUs"):
        BenchmarkConfig.validate_gpu_index("another", 4, 5)


# --- Tests for for_machine (Class Method) ---

def test_for_machine_no_gpu_machine(machines_config, caplog):
    """Test configuration generation for a machine with no GPUs."""
    caplog.set_level(logging.DEBUG)
    config = BenchmarkConfig.for_machine("lima", gpu_index=None, ram_margin=2, vram_margin=0, temperature=0.5)

    assert config.cpu_max_memory == 118  # 120 - 2
    assert config.gpu_max_memory is None # For no GPU machines, vram is null, and it calculates as None
    assert config.gpu_index is None
    assert config.temperature == 0.5 # Test kwargs passing

    assert "Reading config for machine 'lima', of type 'no-gpu'" in caplog.text


@patch('gsm_benchmarker.benchmark_config.torch.cuda.is_available', return_value=True)
def test_for_machine_single_gpu_machine_auto_index(mock_cuda_avail, caplog):
    """Test configuration generation for a single GPU machine with _AUTO index."""
    caplog.set_level(logging.DEBUG)
    config = BenchmarkConfig.for_machine("douro", ram_margin=4, vram_margin=2)

    # Machine: single-v100 (RAM: 64, VRAM: 32, GPUs: 1)
    # CPU: 64 - 4 = 60GiB
    # VRAM: 32 - 2 = 30GiB (Post-init will set index to 0 since CUDA is mocked as available)

    assert config.cpu_max_memory == 60
    assert config.gpu_max_memory == 30
    # Note: The _AUTO resolves to 0 in __post_init__ because torch.cuda.is_available is mocked True
    assert config.gpu_index == 0


def test_for_machine_multi_gpu_machine_explicit_index(machines_config):
    """Test configuration generation for a multi-GPU machine with explicit index."""
    config = BenchmarkConfig.for_machine("cavado", gpu_index=1, ram_margin=10, vram_margin=5)

    # Machine: double-a40 (RAM: 240, VRAM: 48, GPUs: 2)
    # CPU: 240 - 10 = 230GiB
    # VRAM: 48 - 5 = 43GiB

    assert config.cpu_max_memory == 230
    assert config.gpu_max_memory == 43
    assert config.gpu_index == 1
    assert config.max_new_tokens == 1024 # Test default preserved


def test_for_machine_unknown_machine_name():
    """Test ValueError when machine name is not found."""
    with pytest.raises(ValueError, match="No configuration defined for machine name 'unknown_pc'"):
        BenchmarkConfig.for_machine("unknown_pc")


def test_for_machine_invalid_gpu_index_too_high():
    """Test ValueError when the requested GPU index is out of bounds."""
    # cavado has 2 GPUs (indices 0, 1). Trying to use index 2 is invalid.
    with pytest.raises(ValueError, match="Cannot use GPU 2 for machine 'cavado' with a total of 2 GPUs"):
        BenchmarkConfig.for_machine("cavado", gpu_index=2)


def test_for_machine_with_explicit_none_gpu_index(caplog):
    """Test scenario where GPUs are available but user explicitly sets gpu_index=None."""
    caplog.set_level(logging.WARNING)
    config = BenchmarkConfig.for_machine("guadiana", gpu_index=None, ram_margin=5, vram_margin=5)

    # Machine: quadruple-a100 (RAM: 960, VRAM: 80, GPUs: 4)
    # CPU: 960 - 5 = 955GiB
    # VRAM should be None

    assert config.cpu_max_memory == 955
    assert config.gpu_max_memory is None
    assert config.gpu_index is None
    assert "gpu_index is set to None; none of the available 4 GPUs will be used" in caplog.text
