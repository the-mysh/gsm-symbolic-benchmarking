import os
import traceback
from tqdm.auto import tqdm
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from datasets import Dataset

from gsm_benchmarker.benchmark_config import BenchmarkConfig
from gsm_benchmarker.answer_extractor import AnswerExtractor
from gsm_benchmarker.models_config_parser import SingleModelConfig
from gsm_benchmarker.shot_manager import GSM8hotManager
from gsm_benchmarker.utils.path_ops import confirm_or_create_folder, make_name_path_friendly, remove_intermediate_results_folder
from gsm_benchmarker.api_model_wrapper import APIModelWrapper
from gsm_benchmarker.hf_model_wrapper import HFModelWrapper
from gsm_benchmarker.base_model_wrapper import BaseModelWrapper


logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate models using HuggingFace transformers"""

    QUESTION_FORMAT = "Q: {question}\nA: Let's think step by step."
    SHOT_FORMAT = QUESTION_FORMAT + " {solution} The final answer is {result}."

    def __init__(self, model_spec: str | SingleModelConfig, config: BenchmarkConfig):
        self.original_shots = GSM8hotManager()
        self.model_wrapper = self._make_model_wrapper(model_spec, config)

    @staticmethod
    def _make_model_wrapper(model_spec: SingleModelConfig, config: BenchmarkConfig) -> BaseModelWrapper:
        if model_spec.api_type is None:
            logger.debug("Initialising a HuggingFace model")
            return HFModelWrapper(model_spec, config=config)
        else:
            logger.debug("Initialising an API-based model")
            return APIModelWrapper(model_spec, config=config)

    @property
    def model_name(self) -> str:
        return self.model_wrapper.model_name

    @property
    def path_friendly_model_name(self) -> str:
        return make_name_path_friendly(self.model_name)

    def create_prompt(self, question: str) -> str:
        """Create 8-shot CoT prompt following paper's format"""

        sep = "\n\n"

        prompt = "As an expert problem solver, solve step by step the following mathematical questions."
        prompt += sep
        prompt += self.original_shots.format(self.SHOT_FORMAT, separator=sep)
        prompt += sep
        prompt += self.QUESTION_FORMAT.format(question=question)

        return prompt

    def evaluate_dataset(self, dataset: Dataset, leave_progressbar: bool = True) -> pd.DataFrame:
        """
        Evaluate model on a dataset

        Args:
            dataset             :   A Dataset of examples to evaluate.
                                    Required fields: 'question', 'answer', 'numerical_result'.
            leave_progressbar   :   If True, let the shown progress bar remain on the screen after completion.
                                    Otherwise, remove it when done.

        Returns:
            Pandas DataFrame with detailed results, including all columns from the dataset.
        """

        results = []
        prompt_template = self.create_prompt(question="{}")

        for example in tqdm(dataset, desc="Example", leave=leave_progressbar):
            # Extract ground truth answer
            true_result = example['numerical_result']

            if true_result is None:
                logger.warning(f"Could not extract numerical result from: {example['answer']}")
                response, predicted_result, result_pattern, correct = None, None, None, None
            else:
                # Generate prediction
                prompt = prompt_template.format(example['question'])
                response = self.model_wrapper.ask(prompt)
                predicted_result, result_pattern = AnswerExtractor.extract_answer(response)
                correct = predicted_result is not None and abs(predicted_result - true_result) < 1e-5

            results.append({
                **example,
                'predicted_numerical_result': predicted_result,
                'correct': correct,
                'detected_result_pattern': result_pattern.name if result_pattern is not None else None,
                'full_response': response
            })

        return pd.DataFrame(results)

    def evaluate_multiple_datasets(
            self,
            datasets: list[Dataset],
            intermediate_storage_path: Path | str | None = None,
            remove_intermediate_results: bool = True,
            leave_progressbar: bool = True,
        ) -> tuple[pd.DataFrame | None, list[Exception]]:
        """Evaluate model on a set of datasets. Return results in a combined dataframe."""

        if intermediate_storage_path is None:
            logger.debug("No intermediate storage path provided; intermediate results will not be stored")
        else:
            intermediate_storage_path = self._establish_storage_dir(intermediate_storage_path)  # adds a subfolder
            logger.debug(f"Intermediate results will be stored at: {intermediate_storage_path}")

        all_results = []
        n = len(datasets)
        caught_exceptions = []

        for i, dataset in tqdm(enumerate(datasets), desc="Dataset", total=n, leave=leave_progressbar):
            try:
                result = self.evaluate_dataset(dataset, leave_progressbar=False)
                self._store_intermediate_result(result, intermediate_storage_path, i)
                all_results.append(result)
            except Exception as exc:
                caught_exceptions.append(exc)
                logger.error(f"Exception occurred when evaluating dataset {i+1}/{n}: {exc}. "
                             f"Results for this dataset will be skipped.")
                logger.error(f"Full stack:\n{traceback.format_exc()}")

        if all_results:
            full_result = pd.concat(all_results, keys=np.arange(n), names=['set_number', 'question_number'])
        else:
            logger.warning("All evaluations failed, no results to return")
            full_result = None

        if remove_intermediate_results and intermediate_storage_path is not None:
            remove_intermediate_results_folder(intermediate_storage_path)

        return full_result, caught_exceptions

    def _establish_storage_dir(self, storage_path: Path | str) -> Path:
        storage_path = confirm_or_create_folder(storage_path)

        results_folder = f"{self.path_friendly_model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}"
        full_storage_path = storage_path/results_folder
        logger.debug(f"Creating folder for results from current execution: {full_storage_path}")
        os.makedirs(full_storage_path)

        return full_storage_path

    @staticmethod
    def _store_intermediate_result(result: pd.DataFrame, dir_path: Path | None, result_index: int) -> None:
        if dir_path is None:
            return

        fname = dir_path / f"intermediate_{result_index}.parquet"
        if fname.exists():
            # just in case
            logger.warning(f"Intermediate results file {fname} already exists; it will be overwritten")
        result.to_parquet(fname, index=False)
        logger.debug(f"Stored intermediate results at: {fname}")

