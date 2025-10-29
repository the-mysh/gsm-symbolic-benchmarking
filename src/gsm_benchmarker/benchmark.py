import numpy as np
import logging
import pandas as pd
import torch
from pathlib import Path
from functools import cached_property
from dataclasses import dataclass
import traceback

from gsm_benchmarker.dataset_wrapper import GSMSymbolicDataset
from gsm_benchmarker.api_model_wrapper import APIType
from gsm_benchmarker.benchmark_config import BenchmarkConfig
from gsm_benchmarker.model_evaluator import ModelEvaluator
from gsm_benchmarker.utils.path_ops import confirm_or_create_folder, remove_intermediate_results_folder


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
            models: list[str | tuple[str, None | APIType]],
            dset_variants: list[GSMSymbolicDataset.Variant],
            storage_path: Path | str,
            config: BenchmarkConfig | None = None
    ):

        self._models = models
        self._dset_variants = dset_variants
        self._config = config if config is not None else BenchmarkConfig()
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

    def _load_model(self, model: str, api_type: APIType | None = None) -> ModelEvaluator | None:
        try:
            model_evaluator = ModelEvaluator(model, self._config, api_type=api_type)
        except Exception as exc:
            self._handle_model_loading_exception(model, exc)
            return None

        return model_evaluator

    @staticmethod
    def _get_model_and_api_type(model_spec: str | tuple[str, APIType | None]) -> tuple[str, APIType | None]:
        if isinstance(model_spec, str):
            return model_spec, None

        if not isinstance(model_spec, (tuple, list)):
            raise TypeError(f"Expected a model name string or a tuple containing model name and API type; "
                            f"got {type(model_spec)}: {model_spec}")

        if len(model_spec) != 2:
            raise ValueError(f"Expected a tuple of 2 elements; got {len(model_spec)} - {model_spec}")

        model, api_type = model_spec
        if not isinstance(model, str):
            raise TypeError(f"Model name should be a str; got {type(model)}: {model}")

        return model, api_type  # api_type check later

    def run(self, n_sets: int | None = None, n_per_set: int | None = None,
            remove_intermediate_results: bool = True) -> dict[GSMSymbolicDataset.Variant, dict[str, pd.DataFrame]]:
        self._pre_populate_results()

        for im, model_spec in enumerate(self._models):
            model, api_type = self._get_model_and_api_type(model_spec)
            logger.info(f"{10*'='} Evaluating model {im+1}/{len(self._models)}: {model} {10*'='}")

            model_evaluator = self._load_model(model, api_type=api_type)
            if not model_evaluator:
                continue  # failed to load; already handled

            for iv, variant in enumerate(self._dset_variants):
                try:
                    logger.info(f"Evaluating {model} on dataset variant {iv+1}/{len(self._dset_variants)}: {variant}")

                    # Load dataset
                    dataset_wrapper = GSMSymbolicDataset(variant=variant)
                    eval_sets = dataset_wrapper.create_evaluation_sets(n_sets=n_sets, n_per_set=n_per_set)

                    # Evaluate model
                    res, caught_exceptions = model_evaluator.evaluate_multiple_datasets(
                        eval_sets,
                        intermediate_storage_path=self.intermediate_storage_path,
                        remove_intermediate_results=remove_intermediate_results
                    )

                    # Store results
                    self._results[variant][model] = res

                    if res is not None:
                        self._store_model_x_variant_result(dataset_wrapper, model_evaluator, res)
                        
                    if caught_exceptions:
                        # raise an exception to note the failure in the evaluation exceptions summary later on
                        raise RuntimeError(f"Encountered {len(caught_exceptions)} when evaluating {model} on variant {variant}") from caught_exceptions[0]

                    logger.info(f"Evaluation of model {model} on variant {variant} completed")
                except Exception as exc:
                    self._handle_evaluation_exception(model, variant, exc)

            logger.info(f"Evaluation of model {model} on all dataset variants completed")
            self._delete_model(model_evaluator)

        logger.info("EVALUATION COMPLETE")

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

    def _store_model_x_variant_result(self, dataset_wrapper, model_evaluator, res):
        logger.debug("Storing results")

        res_path = self.final_storage_path / dataset_wrapper.path_friendly_name
        confirm_or_create_folder(res_path)

        fname = res_path / f"{model_evaluator.path_friendly_model_name}.parquet"
        res.to_parquet(fname)

        logger.debug(f"Model x variant results stored to: {fname}")

    def summarise_results(self) -> pd.DataFrame:
        summaries = []
        for variant_name, variant_results in self._results.items():
            for model_name, result in variant_results.items():
                correctness = result.loc[result['true_answer'].notna(), 'correct']
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
