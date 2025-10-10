import re
from typing import List, Dict, Optional
from tqdm.auto import tqdm
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from gsm_benchmarker.benchmark_config import BenchmarkConfig



logger = logging.getLogger(__name__)



class AnswerExtractor:
    """Extract numerical answers from model outputs"""

    @staticmethod
    def extract_answer(text: str) -> Optional[float]:
        """
        Extract numerical answer from text.
        Looks for patterns like "#### NUMBER" or "The answer is NUMBER"
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


class HuggingFaceModelEvaluator:
    """Evaluate models using HuggingFace transformers"""

    def __init__(self, model_name: str, config: BenchmarkConfig):
        self.model_name = model_name
        self.config = config
        self.extractor = AnswerExtractor()

        logger.info(f"Loading model: {model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=config.trust_remote_code
        )

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with quantization for efficiency
        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": config.trust_remote_code,
        }

        if config.use_4bit and config.device == "cuda":
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )

        logger.info(f"Model loaded on {config.device}")

    def create_prompt(self, question: str, answer: str = None) -> str:
        """Create 8-shot CoT prompt following paper's format"""

        # Base system instruction
        prompt = "As an expert problem solver, solve step by step the following mathematical questions.\n\n"

        # In practice, you'd load actual 8-shot examples from GSM8K
        # For now, using the question's own answer field if available
        # You should replace this with proper few-shot examples

        prompt += f"Q: {question}\nA: Let's think step by step."

        return prompt

    def generate(self, prompt: str) -> str:
        """Generate response from model"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature if self.config.temperature > 0 else 1.0,
                do_sample=self.config.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the generated part (not the prompt)
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return generated_text

    def evaluate_dataset(self, dataset: List[Dict]) -> Dict:
        """
        Evaluate model on a dataset

        Args:
            dataset: List of examples with 'question' and 'answer' fields

        Returns:
            Dictionary with accuracy and detailed results
        """
        correct = 0
        total = len(dataset)
        results = []

        for example in tqdm(dataset, desc="Evaluating"):
            question = example['question']
            # Extract ground truth answer
            true_answer = self.extractor.extract_answer(example['answer'])

            # Generate prediction
            prompt = self.create_prompt(question)
            response = self.generate(prompt)
            predicted_answer = self.extractor.extract_answer(response)

            is_correct = (
                    predicted_answer is not None and
                    true_answer is not None and
                    abs(predicted_answer - true_answer) < 1e-5
            )

            if is_correct:
                correct += 1

            results.append({
                'question': question,
                'true_answer': true_answer,
                'predicted_answer': predicted_answer,
                'correct': is_correct,
                'response': response
            })

        accuracy = (correct / total * 100) if total > 0 else 0

        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'results': results
        }

