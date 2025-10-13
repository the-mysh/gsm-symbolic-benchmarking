from tqdm.auto import tqdm
import logging
from datasets import load_dataset, arrow_dataset


logger = logging.getLogger(__name__)


class GSMSymbolicDataset:
    """Handler for GSM-Symbolic dataset from HuggingFace"""

    DSET_NAME = "apple/GSM-Symbolic"

    def __init__(self, variant: str = "main"):
        """
        Load GSM-Symbolic dataset

        Args:
            variant: Dataset variant - 'main' (default) / 'p1' / 'p2'.
        """
        logger.info(f"Loading GSM-Symbolic dataset (variant: {variant})...")

        # Load from HuggingFace
        self.dataset = load_dataset(self.DSET_NAME, variant, split="test")

        logger.info(f"Loaded {len(self.dataset)} examples")

    def get_subdataset_for_original_id(self, original_id: int) -> arrow_dataset:
        """Get all instances of a specific question template"""

        return self.dataset.filter(
            lambda x: x == original_id,
            input_columns=["original_id"]
        )

    def get_unique_templates(self) -> list[int]:
        """Get list of unique template IDs"""

        return list(set(self.dataset['original_id']))

    def create_evaluation_sets(self, num_sets: int = 50) -> list[list[dict]]:
        """
        Create 50 evaluation sets (matching paper's methodology)
        Each set contains 100 examples (one per template)

        Returns:
            list of <num_instances> sets, each with 100 examples
        """

        logger.info(f"Creating {num_sets} sets")

        templates = self.get_unique_templates()
        eval_sets = []

        for instance_idx in tqdm(range(num_sets), desc="set", position=0):
            eval_set = []
            for template_id in templates:
                # Get all instances for this template
                sub_dset = self.get_subdataset_for_original_id(template_id)

                # Take the instance_idx-th example (if exists)
                if instance_idx < len(sub_dset):
                    eval_set.append(sub_dset[instance_idx])
                else:
                    # Fallback to first instance if not enough variants
                    eval_set.append(sub_dset[0])

            eval_sets.append(eval_set)

        return eval_sets
