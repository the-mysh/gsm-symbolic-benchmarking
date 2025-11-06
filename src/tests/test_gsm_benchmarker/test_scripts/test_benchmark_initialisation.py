# written with the help of Gemini

import pytest
from unittest.mock import patch, MagicMock
from argparse import Namespace
from pathlib import Path
from datetime import datetime
from enum import Enum

from gsm_benchmarker.scripts.benchmark import (
    choose_models,
    choose_dataset_variants,
    get_paths,
    make_config,
    make_parser,
)


# --- Mocking External Dependencies ---

# 1. Mocking GSMSymbolicDataset.Variant (Enum-like behavior)
class MockVariant(Enum):
    """Mocks the enum for dataset variants."""
    GSM8K = "GSM8K_MOCK"
    main = "MAIN_MOCK"
    p1 = "P1_MOCK"
    p2 = "P2_MOCK"


# 2. Mocking ModelsConfig
class MockModelsConfig:
    """Mocks the configuration object for models."""
    def __init__(self):
        self.open_models = ["model_a_cfg", "model_b_cfg"]

    def __getitem__(self, name):
        """Mocks model lookup."""
        if name == "model1": return "model1_cfg"
        if name == "model2": return "model2_cfg"
        raise KeyError(name)

# 3. Patching imports globally within the module's namespace
@pytest.fixture(autouse=True)
def mock_external_deps():
    """Patches all necessary external imports for isolated testing."""
    with (
        patch('gsm_benchmarker.scripts.benchmark.GSMSymbolicDataset') as MockDset,
        patch('gsm_benchmarker.scripts.benchmark.ModelsConfig', MockModelsConfig),
        patch('gsm_benchmarker.scripts.benchmark.BenchmarkConfig') as MockBenchmarkConfig,
        patch('gsm_benchmarker.scripts.benchmark.socket') as MockSocket
    ):
        # Set up GSMSymbolicDataset.Variant
        MockDset.Variant = MockVariant

        # Set up socket for make_config machine detection
        MockSocket.gethostname.return_value = 'test-machine.local'

        yield MockBenchmarkConfig


# --- Tests for choose_dataset_variants ---

def test_choose_dataset_variants_default():
    """Test returns default set of variants when input list is empty."""
    variants = choose_dataset_variants([])
    expected = [
        MockVariant.GSM8K,
        MockVariant.main,
        MockVariant.p1,
        MockVariant.p2
    ]
    assert variants == expected


def test_choose_dataset_variants_explicit_valid():
    """Test returns correct variants for explicit valid names."""
    variants = choose_dataset_variants(['main', 'p1'])
    expected = [MockVariant.main, MockVariant.p1]
    assert variants == expected


def test_choose_dataset_variants_invalid_name():
    """Test raises ValueError for invalid variant name."""
    with pytest.raises(ValueError, match="'unknown' is not a valid dataset variant"):
        choose_dataset_variants(['main', 'unknown'])


# --- Tests for choose_models ---

def test_choose_models_default():
    """Test returns open_models when model list is empty."""
    models = choose_models([])
    # The MockModelsConfig should be used here
    expected = ["model_a_cfg", "model_b_cfg"]
    assert models == expected


def test_choose_models_explicit_valid():
    """Test returns configurations for explicit valid model names."""
    models = choose_models(['model1', 'model2'])
    expected = ["model1_cfg", "model2_cfg"]
    assert models == expected


def test_choose_models_unrecognised_name():
    """Test raises ValueError for unrecognised model name."""
    with pytest.raises(ValueError, match="Unrecognised model name: 'bad_model'"):
        choose_models(['model1', 'bad_model'])


# --- Tests for get_paths ---

@patch('gsm_benchmarker.scripts.benchmark.Path')
@patch('gsm_benchmarker.scripts.benchmark.datetime')
def test_get_paths_default(mock_datetime, mock_Path):
    """Test default path calculation, mocking datetime and complex Path logic."""

    # 1. Mock datetime for consistent run_folder_name
    mock_datetime.now.return_value = datetime(2023, 10, 27, 12, 30, 0)

    # 2. Define the expected final root path as a REAL Path object.
    # This prevents the final divisions (e.g., / 'logs') from being polluted by MagicMock's chaining.
    final_mocked_root_path = Path("/mock/root/data/gsm-symbolic")

    # 3. Create a mock object that stands for the 6th parent (output_root_path before the final join)
    path_after_traverse = MagicMock(name="PathAfterTraverse")

    # 4. Mock the crucial final join: path_after_traverse / "data/gsm-symbolic"
    # This must return a real PosixPath object.
    path_after_traverse.__truediv__.return_value = final_mocked_root_path

    # 5. Simulate the Path(__file__).resolve().parent(x6) chain
    mock_file = MagicMock(name="MockFile")
    mock_file.resolve.return_value = mock_file

    # We must explicitly set the 6th .parent property to return our controlled mock (path_after_traverse).
    # Since the function accesses the .parent attribute repeatedly, we chain the mocks through that attribute.
    current = mock_file

    # Chain 5 layers of .parent attributes which return new mocks
    for i in range(1, 6):
        current.parent = MagicMock(name=f"Parent{i}")
        current = current.parent

    # The result of the 6th .parent call (current is now the 5th parent) is our controlled mock.
    current.parent = path_after_traverse

    # Set the Path() constructor mock to return the starting mock file
    mock_Path.return_value = mock_file

    # Execute
    logs_path, results_path = get_paths()

    # Assert
    expected_root = final_mocked_root_path
    expected_logs = expected_root / 'logs'
    expected_results = expected_root / "outputs/20231027_123000"

    assert logs_path == expected_logs
    assert results_path == expected_results


def test_get_paths_explicit():
    """Test path calculation when both root and folder name are provided."""
    logs_path, results_path = get_paths(
        output_root_path="/tmp/test_data",
        run_folder_name="manual_run_123"
    )

    expected_root = Path("/tmp/test_data").resolve()
    expected_logs = expected_root / 'logs'
    expected_results = expected_root / "outputs/manual_run_123"

    assert logs_path == expected_logs
    assert results_path == expected_results


# --- Tests for make_config ---

def create_namespace(**kwargs):
    """Helper to create a Namespace object from kwargs."""
    defaults = dict(
        no_machine_preset=False, max_ram=None, ram_margin=None,
        max_vram=None, vram_margin=None, gpu_index=None,
        no_gpu=False
    )

    kwargs = defaults | kwargs

    return Namespace(**kwargs)


# -----------------------------------------------------------
# Scenario 1: --no-machine-preset (Calls BenchmarkConfig())
# -----------------------------------------------------------

def test_make_config_no_preset_with_max_memories(mock_external_deps):
    """Test direct config creation with explicit max memory arguments."""
    pargs = create_namespace(
        no_machine_preset=True,
        max_ram=100,
        max_vram=30,
        gpu_index=1,
    )

    make_config(pargs)
    MockConfig = mock_external_deps

    # Should call the constructor directly with kwargs mapped correctly
    MockConfig.assert_called_once_with(
        trust_remote_code_global=True,
        cpu_max_memory=100, # max_ram is mapped
        gpu_max_memory=30,  # max_vram is mapped
        gpu_index=1,
    )
    MockConfig.for_machine.assert_not_called()


def test_make_config_no_preset_no_gpu(mock_external_deps):
    """Test direct config creation when --no-gpu flag is set."""
    pargs = create_namespace(
        no_machine_preset=True,
        max_ram=50,
        no_gpu=True,
    )

    make_config(pargs)
    MockConfig = mock_external_deps

    MockConfig.assert_called_once_with(
        trust_remote_code_global=True,
        cpu_max_memory=50,
        gpu_index=None, # Should be None due to no_gpu=True
    )


# -----------------------------------------------------------
# Scenario 2: Machine Preset (Calls BenchmarkConfig.for_machine())
# -----------------------------------------------------------

def test_make_config_with_preset_and_margins(mock_external_deps):
    """Test config creation using for_machine with margins and gpu index."""
    # Note: MockSocket.gethostname is already set to 'test-machine.local'
    pargs = create_namespace(
        no_machine_preset=False, # Default
        ram_margin=8,
        vram_margin=4,
        gpu_index=0,
    )

    make_config(pargs)
    MockConfig = mock_external_deps

    # Should call the class method
    MockConfig.for_machine.assert_called_once()

    # Check arguments to for_machine
    call_args, call_kwargs = MockConfig.for_machine.call_args

    # The first argument is the machine name derived from socket.gethostname
    assert call_args[0] == 'test-machine'

    # Check kwargs passed to for_machine
    assert call_kwargs['ram_margin'] == 8
    assert call_kwargs['vram_margin'] == 4
    assert call_kwargs['gpu_index'] == 0
    assert call_kwargs['trust_remote_code_global'] == True

    MockConfig.assert_not_called() # Should not call the constructor


def test_make_config_with_preset_and_max_memories(mock_external_deps):
    """
    Test config creation using for_machine but providing max-ram/vram.
    These should be passed through as arbitrary kwargs to for_machine.
    """
    pargs = create_namespace(
        no_machine_preset=False, # Default
        max_ram=100, # These should be added to kwargs
        max_vram=30, # These should be added to kwargs
        gpu_index=0,
    )

    make_config(pargs)
    MockConfig = mock_external_deps

    # Check arguments to for_machine
    call_args, call_kwargs = MockConfig.for_machine.call_args

    # max_ram/max_vram should be mapped to cpu_max_memory/gpu_max_memory in kwargs
    assert call_kwargs['cpu_max_memory'] == 100
    assert call_kwargs['gpu_max_memory'] == 30

    # Margin args should be absent since they were None in pargs
    assert 'ram_margin' not in call_kwargs
    assert 'vram_margin' not in call_kwargs


def test_make_config_with_preset_no_gpu(mock_external_deps):
    """Test config creation using for_machine when --no-gpu is set."""
    pargs = create_namespace(
        no_machine_preset=False,
        no_gpu=True,
    )

    make_config(pargs)
    MockConfig = mock_external_deps

    # Check arguments to for_machine
    call_args, call_kwargs = MockConfig.for_machine.call_args

    # gpu_index should be explicitly set to None
    assert call_kwargs['gpu_index'] is None
