import numpy as np
import logging
import pandas as pd
import torch
from pathlib import Path
from functools import cached_property
from dataclasses import dataclass
import traceback
from datasets import Dataset
import socket
import json
from time import time

from gsm_benchmarker.dataset_wrapper import GSMSymbolicDataset
from gsm_benchmarker.benchmark_config import BenchmarkConfig
from gsm_benchmarker.models_config_parser import SingleModelConfig
from gsm_benchmarker.model_evaluator import ModelEvaluator
from gsm_benchmarker.utils.path_ops import confirm_or_create_folder, remove_intermediate_results_folder
from gsm_benchmarker.prompt_config import PromptConfig


logger = logging.getLogger(__name__)


@dataclass
class ModelLoadingFailure:
    model: str
    exception: Exception
    exception_traceback: str


@dataclass
class EvaluationFailure:
    model: str
    variant: str
    exception: Exception
    exception_traceback: str


class BenchmarkRunner:
    def __init__(
            self,
            models: list[str | SingleModelConfig],
            dset_variants: list[GSMSymbolicDataset.Variant],
            storage_path: Path | str,
            config: BenchmarkConfig | None = None,
            prompt_config: PromptConfig | None = None
    ):

        self._models = models
        self._dset_variants = dset_variants
        self._config = config if config is not None else BenchmarkConfig()
        self._prompt_config = prompt_config
        self._storage_path = confirm_or_create_folder(storage_path)

        self._results: dict[GSMSymbolicDataset.Variant, dict[str, pd.DataFrame]] = {}
        self._failures: list[EvaluationFailure | ModelLoadingFailure] = []

    @cached_property
    def intermediate_storage_path(self):
        p = self._storage_path / 'intermediate'
        confirm_or_create_folder(p)
        return p

    @cached_property
    def final_storage_path(self):
        p = self._storage_path / 'final'
        confirm_or_create_folder(p)
        return p

    @property
    def results(self) -> dict[GSMSymbolicDataset.Variant, dict[str, pd.DataFrame]]:
        return self._results

    @property
    def failed_evaluations(self) -> list[EvaluationFailure]:
        return self._failures

    def _pre_populate_results(self):
        # results structure: variants as outer layer, models as inner
        # but for efficiency, evaluate per-model (loading models takes longer)
        for variant in self._dset_variants:
            if variant not in self._results:
                self._results[variant] = {}

    def _load_model(self, model_spec: str | SingleModelConfig) -> ModelEvaluator | None:
        try:
            model_evaluator = ModelEvaluator(model_spec, self._config, prompt_config=self._prompt_config)
        except Exception as exc:
            self._handle_model_loading_exception(self._model_name_from_spec(model_spec), exc)
            return None

        return model_evaluator

    @staticmethod
    def _model_name_from_spec(model_spec: str | SingleModelConfig) -> str:
        return model_spec.name if isinstance(model_spec, SingleModelConfig) else model_spec

    def load_datasets(self, n_sets: int | None = None, n_per_set: int | None = None
                      ) -> tuple[dict[GSMSymbolicDataset.Variant, list[Dataset]], dict[GSMSymbolicDataset.Variant, str]]:

        dsets = {}
        dset_names = {}

        logger.debug(f"Loading datasets in {len(self._dset_variants)} variant(s)")

        for variant in self._dset_variants:
            dataset_wrapper = GSMSymbolicDataset(variant=variant)
            dsets[variant] = dataset_wrapper.create_evaluation_sets(n_sets=n_sets, n_per_set=n_per_set)
            dset_names[variant] = dataset_wrapper.path_friendly_name

        return dsets, dset_names

    @staticmethod
    def format_time_diff(td: float):
        minutes, seconds = divmod(td, 60)
        hours, minutes = divmod(minutes, 60)

        return f"{(f"{int(hours)}:" if hours else "")}{int(minutes):02d}:{round(seconds):02d}"

    def run(self, n_sets: int | None = None, n_per_set: int | None = None,
            remove_intermediate_results: bool = True) -> dict[GSMSymbolicDataset.Variant, dict[str, pd.DataFrame]]:

        tt = time()   # t0 for total evaluation time

        self._pre_populate_results()

        dsets, dset_names = self.load_datasets(n_sets, n_per_set)
        self.store_setup_info(n_sets=n_sets, n_per_set=n_per_set)

        for im, model_spec in enumerate(self._models):
            tm = time()  # t0 for model evaluation
            model_name = self._model_name_from_spec(model_spec)
            logger.info(f"{10*'='} Evaluating model {im+1}/{len(self._models)}: {model_name} {10*'='}")

            model_evaluator = self._load_model(model_spec)
            if not model_evaluator:
                continue  # failed to load; already handled

            for iv, variant in enumerate(self._dset_variants):
                tv = time()  # t0 for variant evaluation
                try:
                    logger.info(f"Evaluating {model_name} on dataset variant "
                                f"{iv+1}/{len(self._dset_variants)}: {variant}")

                    # Evaluate model
                    res, caught_exceptions = model_evaluator.evaluate_multiple_datasets(
                        dsets[variant],
                        intermediate_storage_path=self.intermediate_storage_path,
                        remove_intermediate_results=remove_intermediate_results,
                        leave_progressbar=False
                    )

                    # Store results
                    self._results[variant][model_name] = res

                    if res is not None:
                        self._store_model_x_variant_result(dset_names[variant], model_evaluator, res)
                        
                    if caught_exceptions:
                        # raise an exception to note the failure in the evaluation exceptions summary later on
                        raise RuntimeError(f"Encountered {len(caught_exceptions)} error(s() when evaluating "
                                           f"{model_name} on variant {variant}; first one was: {caught_exceptions[0]}")

                    logger.info(f"Evaluation of model {model_name} on variant {variant} completed "
                                f"in {self.format_time_diff(time() - tv)}")
                except Exception as exc:
                    self._handle_evaluation_exception(model_name, variant, exc)

            logger.info(f"Evaluation of model {model_name} on all dataset variants completed "
                        f"in a total of {self.format_time_diff(time() - tm)}")
            self._delete_model(model_evaluator)

        logger.info(f"EVALUATION COMPLETE; total time: {self.format_time_diff(time() - tt)}")

        if remove_intermediate_results:
            remove_intermediate_results_folder(self.intermediate_storage_path)

        logger.info(self.summarise_failures())
        return self._results

    def _handle_evaluation_exception(self, model: str, variant: GSMSymbolicDataset.Variant, exc: Exception):
        t = traceback.format_exc()
        logger.error(f"Evaluation of model {model} on dataset variant {variant.name} failed with an exception: {exc}. "
                     f"Full traceback:\n{t}")
        self._failures.append(EvaluationFailure(
            model=model,
            variant=variant.name,
            exception=exc,
            exception_traceback=t
        ))

    def _handle_model_loading_exception(self, model: str, exc: Exception):
        t = traceback.format_exc()
        logger.error(f"Loading model {model} failed with an exception: {exc}. Full traceback:\n{t}")
        self._failures.append(ModelLoadingFailure(
            model=model,
            exception=exc,
            exception_traceback=t
        ))

    @staticmethod
    def _delete_model(model_evaluator):
        try:
            torch.cuda.empty_cache()
        except Exception as exc:  # expecting AcceleratorError, but couldn't find how to import it
            logger.warning(f"Error emptying CUDA cache: {exc}")
        model_evaluator.model_wrapper.delete_from_cache()

    def store_setup_info(self, **run_params):
        logger.debug("Storing benchmark setup info")

        confirm_or_create_folder(self.final_storage_path)
        fname = self.final_storage_path / 'benchmark_setup_info.json'

        info = dict(
            benchmark_config=self._config.to_dict(),
            run_params=run_params,
            models=[(m.name if isinstance(m, SingleModelConfig) else m) for m in self._models],
            variants=[v.name for v in self._dset_variants],
            machine=socket.gethostname()
        )

        with open(fname, 'w') as f:
            json.dump(info, f, indent=4)
        logger.debug(f"Benchmark setup info stored to: {fname}")

    def _store_model_x_variant_result(self, dset_variant_name, model_evaluator, res):
        logger.debug("Storing results")

        res_path = self.final_storage_path / dset_variant_name
        confirm_or_create_folder(res_path)

        fname = res_path / f"{model_evaluator.path_friendly_model_name}.parquet"
        res.to_parquet(fname)

        logger.debug(f"Model x variant results stored to: {fname}")

    def summarise_results(self) -> pd.DataFrame:
        summaries = []
        for variant_name, variant_results in self._results.items():
            for model_name, result in variant_results.items():
                correctness = result.loc[result['numerical_result'].notna(), 'correct']
                summaries.append(
                    {
                        'variant': variant_name,
                        'model': model_name,
                        'accuracy': np.nanmean(correctness.astype(float))
                    }
                )

        return pd.DataFrame(summaries)

    def summarise_failures(self) -> str:
        if not self._failures:
            return "No evaluation/model loading failures recorded"

        def make_msg(f: ModelLoadingFailure | EvaluationFailure):
            if isinstance(f, ModelLoadingFailure):
                return f"On model {f.model} - failed to load: {f.exception}"
            else:
                return f"On model {f.model}, variant {f.variant} - failed to evaluate: {f.exception}"

        ss = "\n\n".join(make_msg(ef) for ef in self._failures)

        return f"Recorded {len(self._failures)} failures:\n\n{ss}"
