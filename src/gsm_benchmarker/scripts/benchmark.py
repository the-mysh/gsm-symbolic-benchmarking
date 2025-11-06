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

from gsm_benchmarker.dataset_wrapper import GSMSymbolicDataset
from gsm_benchmarker.benchmark_config import BenchmarkConfig
from gsm_benchmarker.benchmark import BenchmarkRunner
from gsm_benchmarker.models_config_parser import ModelsConfig
from gsm_benchmarker.utils.logging_setup import install_colored_logger, setup_log_file_handler
from gsm_benchmarker.utils.seeds import set_seed


logger = logging.getLogger(__name__)


def setup_logs(logs_path, log_level=logging.INFO):
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
    logger.info(f"Run results will be stored to: {results_path}")

    return output_root_path / 'logs', results_path


def make_config(pargs: Namespace):
    kwargs: dict[str, Any] = dict(trust_remote_code_global=True)

    def add_to_kwargs(name, new_name=None):
        if (value := getattr(pargs, name, None)) is not None:
            kwargs[new_name or name] = value

    add_to_kwargs('max_ram', 'cpu_max_memory')
    add_to_kwargs('max_vram', 'gpu_max_memory')

    if getattr(pargs, 'no_gpu', False):
        kwargs['gpu_index'] = None
    else:
        add_to_kwargs('gpu_index')

    if pargs.no_machine_preset:
        bc = BenchmarkConfig(**kwargs)
    else:
        machine = socket.gethostname().split('.')[0]
        logger.info(f"Detected machine: {machine}")

        add_to_kwargs('ram_margin')
        add_to_kwargs('vram_margin')
        bc = BenchmarkConfig.for_machine(machine, **kwargs)

    return bc


def make_parser() -> ArgumentParser:
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

    parser.add_argument('--log-level', default=logging.INFO)
    parser.add_argument('--output-root-path', default=None)
    parser.add_argument('--run-folder-name', default=None)

    parser.add_argument('--variants', nargs='+', choices=['main', 'GSM8K', 'p1', 'p2'])
    parser.add_argument('--models', type=str, nargs='+')

    parser.add_argument('--n-sets', type=int, default=None)
    parser.add_argument('--n-per-set', type=int, default=None)

    return parser


def main():
    pargs = make_parser().parse_args()

    logs_path, results_path = get_paths(pargs.output_root_path, pargs.run_folder_name)
    setup_logs(logs_path, log_level=pargs.log_level)

    set_seed(42)
    hf_login()

    bc = make_config(pargs)
    logger.info(f"Configuration: {bc}")
    br = BenchmarkRunner(
        models=choose_models(pargs.models),
        dset_variants=choose_dataset_variants(pargs.variants),
        storage_path=results_path,
        config=make_config(pargs)
    )

    br.run(n_sets=pargs.n_sets, n_per_set=pargs.n_per_set)

    print(br.summarise_failures())


if __name__ == '__main__':
    main()
