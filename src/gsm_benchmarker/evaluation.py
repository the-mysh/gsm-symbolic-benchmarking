import os
import re
import traceback
from tqdm.auto import tqdm
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from gsm_benchmarker.benchmark_config import BenchmarkConfig
from gsm_benchmarker.model_wrapper import HFModelWrapper
from gsm_benchmarker.shot_manager import GSM8hotManager


logger = logging.getLogger(__name__)


class HuggingFaceModelEvaluator:
    """Evaluate models using HuggingFace transformers"""

    QUESTION_FORMAT = "Q: {question}\nA: Let's think step by step."
    SHOT_FORMAT = QUESTION_FORMAT + " {solution} The final answer is {result}."

    def __init__(self, model_name: str, config: BenchmarkConfig):
        self.original_shots = GSM8hotManager()
        self.model_wrapper = HFModelWrapper(model_name, config=config)

    def create_prompt(self, question: str) -> str:
        """Create 8-shot CoT prompt following paper's format"""

        sep = "\n\n"

        prompt = "As an expert problem solver, solve step by step the following mathematical questions."
        prompt += sep
        prompt += self.original_shots.format(self.SHOT_FORMAT, separator=sep)
        prompt += sep
        prompt += self.QUESTION_FORMAT.format(question=question)

        return prompt

    @staticmethod
    def extract_answer(text: str) -> float | None:
        """
        Extract numerical answer from text.
        Looks for patterns like "#### NUMBER" or "The (final) answer is NUMBER"
        """

        # GSM8K standard format: #### NUMBER
        match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', text)
        if match:
            return float(match.group(1))

        # Alternative formats
        patterns = [
            r'[Tt]he (?:final )?answer is\s*\$?\s*(-?\d+(?:\.\d+)?)',
            r'[Aa]nswer:\s*\$?\s*(-?\d+(?:\.\d+)?)',
            r'=\s*(-?\d+(?:\.\d+)?)\s*(?:\n|$)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return float(match.group(1))

        # Last resort: find last number in text
        numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
        if numbers:
            return float(numbers[-1])

        return None

    def evaluate_dataset(self, dataset: list[dict]) -> pd.DataFrame:
        """
        Evaluate model on a dataset

        Args:
            dataset: list of examples with 'question' and 'answer' fields

        Returns:
            Dictionary with accuracy and detailed results
        """

        results = []
        prompt_template = self.create_prompt(question="{}")

        for example in tqdm(dataset, desc="Example"):
            question = example['question']

            # Extract ground truth answer
            true_answer = self.extract_answer(example['answer'])

            if true_answer is None:
                logger.warning(f"Could not extract numerical answer from: {example['answer']}")
                response, predicted_answer, correct = None, None, None
            else:
                # Generate prediction
                prompt = prompt_template.format(question)
                response = self.model_wrapper.ask(prompt)
                predicted_answer = self.extract_answer(response)
                correct = predicted_answer is not None and abs(predicted_answer - true_answer) < 1e-5

            results.append({
                'id': example['id'],
                'question': question,
                'true_answer': true_answer,
                'predicted_answer': predicted_answer,
                'correct': correct,
                'response': response
            })

        return pd.DataFrame(results)

    def evaluate_multiple_datasets(self, datasets: list[list[dict]], intermediate_storage_path: Path | str | None
                                   ) -> pd.DataFrame:
        """Evaluate model on a set of datasets. Return results in a combined dataframe."""

        if intermediate_storage_path is None:
            logger.info("No intermediate storage path provided; intermediate results will not be stored")
        else:
            intermediate_storage_path = self._establish_storage_dir(intermediate_storage_path)  # adds a subfolder
            logger.info(f"Intermediate results will be stored at: {intermediate_storage_path}")

        all_results = []
        n = len(datasets)

        for i, dataset in tqdm(enumerate(datasets), desc="Dataset"):
            try:
                result = self.evaluate_dataset(dataset)
                self._store_intermediate_result(result, intermediate_storage_path, i)
                all_results.append(result)
            except Exception as exc:
                logger.error(f"Exception occurred when evaluating dataset {i+1}/{n}: {exc}. "
                             f"Results for this dataset will be skipped.")
                logger.error(f"Full stack:\n{traceback.format_exc()}")

        return pd.concat(all_results, keys=np.arange(n))

    @staticmethod
    def _establish_storage_dir(storage_path: Path | str) -> Path:
        if isinstance(storage_path, str):
            storage_path = Path(storage_path)

        if not isinstance(storage_path, Path):
            raise TypeError(f"Expected a str or a Path object; got {type(storage_path)}: {storage_path}")

        if storage_path.is_file():
            raise RuntimeError(f'Expected a path to a directory; got a file path instead: {storage_path}')

        if not storage_path.exists():
            if not storage_path.parent.exists():
                logger.warning(f"Storage root does not exist and will be created")
            logger.info(f"Creating storage directories at {storage_path}")
            os.makedirs(storage_path)

        results_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
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
