import re
from typing import List, Dict, Optional
from tqdm.auto import tqdm
import logging
import pandas as pd

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
    def extract_answer(text: str) -> Optional[float]:
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

    def evaluate_dataset(self, dataset: List[Dict]) -> pd.DataFrame:
        """
        Evaluate model on a dataset

        Args:
            dataset: List of examples with 'question' and 'answer' fields

        Returns:
            Dictionary with accuracy and detailed results
        """

        results = []
        prompt_template = self.create_prompt(question="{}")

        for example in tqdm(dataset, desc="Evaluating"):
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
