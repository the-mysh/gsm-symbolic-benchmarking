import numpy as np
import logging
import pandas as pd
import torch
from pathlib import Path
from functools import cached_property

from gsm_benchmarker.dataset_wrapper import GSMSymbolicDataset
from gsm_benchmarker.benchmark_config import BenchmarkConfig
from gsm_benchmarker.hf_model_evaluator import HuggingFaceModelEvaluator
from gsm_benchmarker.utils.path_ops import confirm_or_create_folder


logger = logging.getLogger(__name__)


class BenchmarkRunner:
    def __init__(self, models: list[str], dset_variants: list[GSMSymbolicDataset.Variant], storage_path: Path | str,
                 config: BenchmarkConfig | None = None):

        self._models = models
        self._dset_variants = dset_variants
        self._config = config if config is not None else BenchmarkConfig()
        self._storage_path = confirm_or_create_folder(storage_path)

        self._results = {}

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

    def _pre_populate_results(self):
        # results structure: variants as outer layer, models as inner
        # but for efficiency, evaluate per-model (loading models takes longer)
        for variant in self._dset_variants:
            if variant not in self._results:
                self._results[variant] = {}

    def run(self, n_sets: int | None = None, n_per_set: int | None = None):
        self._pre_populate_results()

        for im, model in enumerate(self._models):
            logger.info(f"{10*'='} Evaluating model {im+1}/{len(self._models)}: {model} {10*'='}")
            model_evaluator = HuggingFaceModelEvaluator(model, self._config)

            for iv, variant in enumerate(self._dset_variants):
                logger.info(f"Evaluating {model} on dataset variant {iv+1}/{len(self._dset_variants)}: {variant}")

                # Load dataset
                dataset_wrapper = GSMSymbolicDataset(variant=variant)
                eval_sets = dataset_wrapper.create_evaluation_sets(n_sets=n_sets, n_per_set=n_per_set)

                # Evaluate model
                res = model_evaluator.evaluate_multiple_datasets(
                    eval_sets,
                    intermediate_storage_path=self.intermediate_storage_path / f"{model_evaluator.path_friendly_model_name}"
                )

                # Store results
                self._results[variant][model] = res
                self._store_model_x_variant_result(dataset_wrapper, model_evaluator, res)

                logger.info(f"Evaluation of model {model} on variant {variant} completed")

            logger.info(f"Evaluation of model {model} on all dataset variants completed")
            self._delete_model(model_evaluator)

        logger.info("EVALUATION COMPLETE")
        return self._results

    @staticmethod
    def _delete_model(model_evaluator):
        logger.debug("Deleting model from memory")
        del model_evaluator.model_wrapper
        torch.cuda.empty_cache()

    def _store_model_x_variant_result(self, dataset_wrapper, model_evaluator, res):
        logger.debug("Storing results")

        res_path = self.final_storage_path / f"results/{dataset_wrapper.DSET_NAME}_{dataset_wrapper.variant_name}"
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
