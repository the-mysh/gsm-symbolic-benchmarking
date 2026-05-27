"""CLI utilities and entry point for running GSM-Symbolic benchmarks.

This module contains helper functions used by the benchmark runner script
(`BenchmarkRunner`) together with a command-line entry point that parses
arguments, configures logging, and launches a benchmark run.

Functions exposed here are small helpers for logging setup, HuggingFace
login, model/dataset selection, and configuration construction.
"""

import os
from datetime import datetime
from pathlib import Path
import logging
from huggingface_hub import login, whoami
from transformers.utils.logging import disable_progress_bar
import datasets
import socket
from argparse import ArgumentParser, Namespace
from typing import Any

from gsm_benchmarker.input_data_management.dataset_wrapper import GSMSymbolicDataset
from gsm_benchmarker.benchmark.benchmark_config import BenchmarkConfig
from gsm_benchmarker.benchmark.benchmark import BenchmarkRunner
from gsm_benchmarker.model_wrappers.models_config_parser import ModelsConfig
from gsm_benchmarker.utils.logging_setup import install_colored_logger, setup_log_file_handler
from gsm_benchmarker.utils.seeds import set_seed
from gsm_benchmarker.input_data_management.prompt_config import PromptConfig


logger = logging.getLogger(__name__)


def setup_logs(logs_path, log_level=logging.INFO):
    """Configure logging for the benchmark run.

    This silences noisy third-party loggers, installs a coloured logger for
    console output and configures the dataset/transformers progress bars.

    Parameters
    ----------
    logs_path:
        Path where run logs should be written (used by the project's
        file-based handler).
    log_level:
        Logging level for console output (defaults to logging.INFO).
    """

    for log_name in (
            'urllib3', 'fsspec', 'filelock', 'h5py', 'httpcore', 'httpx', 'google_genai', 'jax',
            'root', 'bitsandbytes', 'transformers_modules'
    ):
        logging.getLogger(log_name).setLevel(logging.WARNING)

    if isinstance(log_level, str):
        try:
            log_level = int(log_level)
        except ValueError:
            pass

    install_colored_logger(level=log_level)

    disable_progress_bar()
    datasets.disable_progress_bars()

    setup_log_file_handler(logs_path)


def hf_login():
    """Log in to the HuggingFace Hub using the HUGGINGFACEHUB_API_TOKEN env var.

    Raises
    ------
    RuntimeError
        If the environment variable is not set.
    """

    t = 'HUGGINGFACEHUB_API_TOKEN'
    hf_api_token = os.environ.get(t, None)
    if hf_api_token is None:
        raise RuntimeError(f"{t} is not set; cannot log in to Huggingface Hub")

    login(hf_api_token)
    logger.info(f"Login to Huggingface Hub successful; logged-in user: {whoami()['name']}")

    # check hf cache dir
    hf_home = os.environ.get("HF_HOME", None)
    if hf_home is None:
        logger.warning("HF_HOME is not set")
    else:
        logger.debug(f"HF_HOME is set to {hf_home}")


def choose_models(model_names: list[str]):
    """Resolve model names into configured model objects.

    If `model_names` is empty or None, returns all "open" models from the
    project's `ModelsConfig`. Otherwise validates each requested name and
    returns a list of model configuration objects.
    """

    models_config = ModelsConfig()

    if not model_names:
        return models_config.open_models

    models = []
    for m in model_names:
        try:
            models.append(models_config[m])
        except KeyError:
            raise ValueError(f"Unrecognised model name: '{m}'")
    return models


def choose_dataset_variants(variant_names: list[str]):
    """Return a list of dataset variant enum members from requested names.

    If `variant_names` is empty, returns the default set used by the
    benchmark (GSM8K, main, p1, p2). Validates names against the
    `GSMSymbolicDataset.Variant` enumeration and raises on unknown values.
    """

    vs = GSMSymbolicDataset.Variant

    if not variant_names:
        return [vs.GSM8K, vs.main, vs.p1, vs.p2]

    variants = []
    for v in variant_names:
        try:
            variants.append(vs[v])
        except KeyError:
            raise ValueError(f"'{v}' is not a valid dataset variant; choose from: {', '.join(vs.__members__)}")
    return variants

def get_paths(output_root_path: str | Path | None = None, run_folder_name: str | None = None):
    """Return (logs_path, results_path) for the run.

    If `output_root_path` is omitted, the function infers a project-local
    `data/gsm-symbolic` directory by walking up from this file. The
    `run_folder_name` defaults to a timestamped folder.
    """

    if output_root_path is None:
        output_root_path = Path(__file__).resolve()
        for i in range(6):
            output_root_path = output_root_path.parent
        output_root_path = output_root_path / "data/gsm-symbolic"
    else:
        output_root_path = Path(output_root_path).resolve()

    if run_folder_name is None:
        run_folder_name = datetime.now().strftime('%Y%m%d_%H%M%S')

    results_path = output_root_path / f"outputs/{run_folder_name}"

    return output_root_path / 'logs', results_path


def make_config(pargs: Namespace):
    """Build a `BenchmarkConfig` from parsed CLI arguments.

    The function translates CLI names into the parameter names expected by
    `BenchmarkConfig`, handles GPU selection flags and optionally loads a
    machine preset.
    """

    kwargs: dict[str, Any] = dict(trust_remote_code_global=True)

    def add_to_kwargs(name, new_name=None):
        if (value := getattr(pargs, name, None)) is not None:
            kwargs[new_name or name] = value

    add_to_kwargs('max_ram', 'cpu_max_memory')
    add_to_kwargs('max_vram', 'gpu_max_memory')

    if getattr(pargs, 'no_gpu', False):
        kwargs['gpu_index'] = None
    elif getattr(pargs, 'auto_gpu', False):
        kwargs['gpu_auto'] = True
    else:
        add_to_kwargs('gpu_index')

    kwargs['use_4bit'] = getattr(pargs, 'quantise', False)

    if pargs.no_machine_preset:
        bc = BenchmarkConfig(**kwargs)
    else:
        machine = socket.gethostname().split('.')[0]
        logger.info(f"Detected machine: {machine}")

        add_to_kwargs('ram_margin')
        add_to_kwargs('vram_margin')
        bc = BenchmarkConfig.for_machine(machine, **kwargs)

    logger.info(f"Benchmark configuration: {bc}")

    return bc


def make_prompt_config(preset_name: str | None = None, file_path: str | None = None) -> PromptConfig:
    """Load or build a `PromptConfig`.

    Preference order: explicit `file_path` &gt; named `preset_name` &gt; the
    package default. The prompt configuration is logged with an example
    invocation for quick sanity checks.
    """

    if file_path is not None:
        pc = PromptConfig.from_file(file_path)

    elif preset_name is not None:
        pc = PromptConfig.from_preset(preset_name)

    else:
        pc = PromptConfig.default()

    logger.info(f"Example prompt according to loaded format:\n{pc("<Question here>")}")

    return pc


def make_parser() -> ArgumentParser:
    """Return an ArgumentParser configured for the benchmark CLI.

    The parser mirrors options used by the original benchmark scripts such
    as machine presets, resource limits, selected models/variants and prompt
    format selection.
    """

    parser = ArgumentParser("GSM-Symbolic Benchmark Reproduction")
    parser.add_argument('--no-machine-preset', dest='no_machine_preset', action='store_true', default=False)

    gc = parser.add_mutually_exclusive_group()
    gc.add_argument('--max-ram', type=int, default=None)
    gc.add_argument('--ram-margin', type=int, default=None)

    gg = parser.add_mutually_exclusive_group()
    gg.add_argument('--max-vram', type=int, default=None)
    gg.add_argument('--vram-margin', type=int, default=None)

    g = parser.add_mutually_exclusive_group()
    g.add_argument('--gpu-index', type=int)
    g.add_argument('--no-gpu', dest='no_gpu', action='store_true')
    g.add_argument('--auto-gpu', action='store_true', dest='auto_gpu')

    parser.add_argument('--log-level', default=logging.INFO)
    parser.add_argument('--output-root-path', default=None)
    parser.add_argument('--run-folder-name', default=None)

    parser.add_argument('--variants', nargs='+', choices=['main', 'GSM8K', 'p1', 'p2'])
    parser.add_argument('--models', type=str, nargs='+')

    parser.add_argument('--n-sets', type=int, default=None)
    parser.add_argument('--n-per-set', type=int, default=None)

    parser.add_argument('--quantise', action='store_true', default=False)

    gp = parser.add_mutually_exclusive_group()
    gp.add_argument('--prompt-preset', default=None)
    gp.add_argument('--prompt-format-file')

    return parser


def main():
    pargs = make_parser().parse_args()

    logs_path, results_path = get_paths(pargs.output_root_path, pargs.run_folder_name)
    setup_logs(logs_path, log_level=pargs.log_level)

    set_seed(42)
    hf_login()

    logger.info(f"Run results will be stored to: {results_path}")

    bc = make_config(pargs)

    pc = make_prompt_config(
        preset_name=getattr(pargs, 'prompt_preset', None),
        file_path=getattr(pargs, 'prompt_format_file', None)
    )

    br = BenchmarkRunner(
        models=choose_models(pargs.models),
        dset_variants=choose_dataset_variants(pargs.variants),
        storage_path=results_path,
        config=bc,
        prompt_config=pc
    )

    br.run(n_sets=pargs.n_sets, n_per_set=pargs.n_per_set)

    print(br.summarise_failures())


if __name__ == '__main__':
    main()
