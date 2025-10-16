import numpy as np
import logging
import os
import pandas as pd
import torch
from pathlib import Path

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

    def run(self, n_sets: int | None = None, n_per_set: int | None = None):
        # TODO: refactor

        for variant in self._dset_variants:
            if variant not in self._results:
                self._results[variant] = {}

        intermediate_storage_path = self._storage_path/'intermediate'
        final_storage_path = self._storage_path/'results'

        for im, model in enumerate(self._models):
            logger.info(f"{10*'='} Evaluating model {im+1}/{len(self._models)}: {model} {10*'='}")
            model_evaluator = HuggingFaceModelEvaluator(model, self._config)

            for iv, variant in enumerate(self._dset_variants):
                logger.info(f"Evaluating {model} on dataset variant {iv+1}/{len(self._dset_variants)}: {variant}")

                # Load dataset
                dataset_wrapper = GSMSymbolicDataset(variant=variant)
                eval_sets = dataset_wrapper.create_evaluation_sets(n_sets=n_sets, n_per_set=n_per_set)

                res = model_evaluator.evaluate_multiple_datasets(
                    eval_sets,
                    intermediate_storage_path=intermediate_storage_path / f"{model_evaluator.path_friendly_model_name}"
                )

                logger.debug("Storing results")
                self._results[variant][model] = res

                res_path = final_storage_path / f"results/{dataset_wrapper.DSET_NAME}_{dataset_wrapper.variant_name}"
                os.makedirs(res_path, exist_ok=True)
                fname = res_path / f"{model_evaluator.path_friendly_model_name}.parquet"
                res.to_parquet(fname)
                logger.debug(f"Model x variant results stored to: {fname}")

                logger.info(f"Evaluation of model {model} on variant {variant} completed")

            logger.info(f"Evaluation of model {model} on all dataset variants completed")
            logger.debug("Deleting model from memory")
            del model_evaluator.model_wrapper
            torch.cuda.empty_cache()

        logger.info("EVALUATION COMPLETE")
        return self._results


def run_full_benchmark():
    """
    Run complete benchmark reproducing Appendix A.2
    """
    config = BenchmarkConfig()

    # Define models to evaluate
    models_config = {
        # HuggingFace models
        'hf': [
            "meta-llama/Llama-3.1-8B",
            # "meta-llama/Llama-3-8b-instruct",
            # "google/gemma-2-9b-it",
            # "microsoft/Phi-3-mini-128k-instruct",
            # Add more models as needed
        ],
        # API-based models (if available)
        'openai': [
            # "gpt-4o-mini",
            # "gpt-4o",
        ],
        'anthropic': [
            # "claude-sonnet-4-5-20250929",
        ]
    }

    # Dataset variants from paper
    variants = {
        'GSM8K-subset': None,  # Original 100 questions
        'GSM-Symbolic-M1': 'm1',
        'GSM-Symbolic': 'p1',  # Default
        'GSM-Symbolic-P1': 'p1',
        'GSM-Symbolic-P2': 'p2',
        'GSM-NoOp': 'noop',
    }

    results_table = []

    # Evaluate each model
    for model_type, model_list in models_config.items():
        for model_name in model_list:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Evaluating: {model_name}")
            logger.info(f"{'=' * 60}\n")

            model_results = {'Model': model_name}

            # Initialize evaluator
            if model_type == 'hf':
                evaluator = HuggingFaceModelEvaluator(model_name, config)
            else:
                raise NotImplementedError("API evaluator not implemented")

            # Test on each variant
            for variant_name, variant_split in variants.items():
                if variant_split is None:
                    continue  # Skip if not implemented

                logger.info(f"\nVariant: {variant_name}")

                # Load dataset
                dataset_loader = GSMSymbolicDataset(variant_split)
                eval_sets = dataset_loader.create_evaluation_sets(n_sets=50)

                # Evaluate on all 50 instances
                accuracies = []
                for i, eval_set in enumerate(eval_sets):
                    result = evaluator.evaluate_dataset(eval_set)
                    accuracies.append(result['accuracy'])
                    logger.info(f"  Instance {i + 1}/50: {result['accuracy']:.1f}%")

                # Compute statistics
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)

                model_results[variant_name] = f"{mean_acc:.1f} ({std_acc:.2f})"
                logger.info(f"\n  Mean: {mean_acc:.1f}%, Std: {std_acc:.2f}%")

            results_table.append(model_results)

            # Clean up model to free memory
            if model_type == 'hf':
                del evaluator.model
                torch.cuda.empty_cache()

    # Save results
    df = pd.DataFrame(results_table)
    output_path = 'results/gsm_symbolic_appendix_a2.csv'
    os.makedirs('results', exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"\n{'=' * 60}")
    logger.info("FINAL RESULTS")
    logger.info(f"{'=' * 60}\n")
    logger.info(df.to_string(index=False))
    logger.info(f"\nResults saved to: {output_path}")

    return df


if __name__ == '__main__':
    run_full_benchmark()
