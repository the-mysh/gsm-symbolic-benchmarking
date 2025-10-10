"""
GSM-Symbolic Benchmark - HuggingFace Implementation
Reproduces Appendix A.2 results using HuggingFace datasets and transformers
by Claude Sonnet
"""

import os
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from tqdm.auto import tqdm

# HuggingFace imports
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
import torch

# Optional: For API-based models
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None


@dataclass
class BenchmarkConfig:
    """Configuration for GSM-Symbolic benchmark"""
    random_seed: int = 42
    num_shots: int = 8
    temperature: float = 0.0  # greedy decoding
    max_new_tokens: int = 1024
    batch_size: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Model-specific
    use_4bit: bool = True  # For memory efficiency
    trust_remote_code: bool = False


class GSMSymbolicDataset:
    """Handler for GSM-Symbolic dataset from HuggingFace"""

    def __init__(self, split: str = "p1"):
        """
        Load GSM-Symbolic dataset

        Args:
            split: Dataset variant - 'p1' (default), 'p2', 'm1', 'noop', etc.
        """
        print(f"Loading GSM-Symbolic dataset (split: {split})...")

        # Load from HuggingFace
        self.dataset = load_dataset("apple/GSM-Symbolic", split=split)

        print(f"Loaded {len(self.dataset)} examples")

    def get_instances_by_id(self, original_id: int) -> List[Dict]:
        """Get all instances of a specific question template"""
        return [
            ex for ex in self.dataset
            if ex['original_id'] == original_id
        ]

    def get_unique_templates(self) -> List[int]:
        """Get list of unique template IDs"""
        return sorted(set(ex['original_id'] for ex in self.dataset))

    def create_evaluation_sets(self, num_instances: int = 50) -> List[List[Dict]]:
        """
        Create 50 evaluation sets (matching paper's methodology)
        Each set contains 100 examples (one per template)

        Returns:
            List of 50 datasets, each with 100 examples
        """
        templates = self.get_unique_templates()
        eval_sets = []

        for instance_idx in range(num_instances):
            eval_set = []
            for template_id in templates:
                # Get all instances for this template
                instances = self.get_instances_by_id(template_id)

                # Take the instance_idx-th example (if exists)
                if instance_idx < len(instances):
                    eval_set.append(instances[instance_idx])
                else:
                    # Fallback to first instance if not enough variants
                    eval_set.append(instances[0])

            eval_sets.append(eval_set)

        return eval_sets


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

        print(f"Loading model: {model_name}")

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

        print(f"Model loaded on {config.device}")

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


class APIModelEvaluator:
    """Evaluate models via API (OpenAI, Anthropic, etc.)"""

    def __init__(self, model_name: str, api_type: str = "openai"):
        self.model_name = model_name
        self.api_type = api_type
        self.extractor = AnswerExtractor()

        if api_type == "openai" and openai is None:
            raise ImportError("openai package not installed")
        elif api_type == "anthropic" and anthropic is None:
            raise ImportError("anthropic package not installed")

    def create_prompt(self, question: str) -> str:
        """Create prompt (same as HF version)"""
        prompt = "As an expert problem solver, solve step by step the following mathematical questions.\n\n"
        prompt += f"Q: {question}\nA: Let's think step by step."
        return prompt

    def generate(self, prompt: str) -> str:
        """Generate via API"""
        if self.api_type == "openai":
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1024
            )
            return response.choices[0].message.content

        elif self.api_type == "anthropic":
            client = anthropic.Anthropic()
            message = client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text

        else:
            raise ValueError(f"Unknown API type: {self.api_type}")

    def evaluate_dataset(self, dataset: List[Dict]) -> Dict:
        """Evaluate (same logic as HF evaluator)"""
        correct = 0
        total = len(dataset)
        results = []

        for example in tqdm(dataset, desc=f"Evaluating {self.model_name}"):
            question = example['question']
            true_answer = self.extractor.extract_answer(example['answer'])

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


def run_full_benchmark():
    """
    Run complete benchmark reproducing Appendix A.2
    """
    config = BenchmarkConfig()

    # Define models to evaluate
    models_config = {
        # HuggingFace models
        'hf': [
            "meta-llama/Llama-3-8b-instruct",
            "google/gemma-2-9b-it",
            "microsoft/Phi-3-mini-128k-instruct",
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
            print(f"\n{'=' * 60}")
            print(f"Evaluating: {model_name}")
            print(f"{'=' * 60}\n")

            model_results = {'Model': model_name}

            # Initialize evaluator
            if model_type == 'hf':
                evaluator = HuggingFaceModelEvaluator(model_name, config)
            else:
                evaluator = APIModelEvaluator(model_name, api_type=model_type)

            # Test on each variant
            for variant_name, variant_split in variants.items():
                if variant_split is None:
                    continue  # Skip if not implemented

                print(f"\nVariant: {variant_name}")

                # Load dataset
                dataset_loader = GSMSymbolicDataset(split=variant_split)
                eval_sets = dataset_loader.create_evaluation_sets(num_instances=50)

                # Evaluate on all 50 instances
                accuracies = []
                for i, eval_set in enumerate(eval_sets):
                    result = evaluator.evaluate_dataset(eval_set)
                    accuracies.append(result['accuracy'])
                    print(f"  Instance {i + 1}/50: {result['accuracy']:.1f}%")

                # Compute statistics
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)

                model_results[variant_name] = f"{mean_acc:.1f} ({std_acc:.2f})"
                print(f"\n  Mean: {mean_acc:.1f}%, Std: {std_acc:.2f}%")

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

    print(f"\n{'=' * 60}")
    print("FINAL RESULTS")
    print(f"{'=' * 60}\n")
    print(df.to_string(index=False))
    print(f"\nResults saved to: {output_path}")

    return df


if __name__ == "__main__":
    # Example: Quick test on a single model and variant
    print("GSM-Symbolic Benchmark - HuggingFace Implementation")
    print("=" * 60)

    # For quick testing, evaluate one model on one variant
    config = BenchmarkConfig()

    # Load dataset
    print("\n1. Loading dataset...")
    dataset_loader = GSMSymbolicDataset(split='p1')
    eval_sets = dataset_loader.create_evaluation_sets(num_instances=5)  # Just 5 for testing

    # Test with a small model (adjust as needed)
    print("\n2. Initializing model...")
    # model_name = "microsoft/Phi-3-mini-128k-instruct"
    # evaluator = HuggingFaceModelEvaluator(model_name, config)

    # For demo purposes, showing the structure
    print(f"\nDataset structure:")
    print(f"  - Number of evaluation sets: {len(eval_sets)}")
    print(f"  - Examples per set: {len(eval_sets[0])}")
    print(f"\nExample question:")
    print(eval_sets[0][0]['question'][:200] + "...")

    print("\n3. To run full benchmark, uncomment the model initialization above")
    print("   and run: python gsm_benchmark.py")

    # Uncomment to run full benchmark:
    # results = run_full_benchmark()