from tqdm.auto import tqdm
import logging
from datasets import load_dataset, Dataset
from enum import Enum, auto
from typing import TypeVar

from gsm_benchmarker.utils.path_ops import make_name_path_friendly

T = TypeVar("T")

logger = logging.getLogger(__name__)


class GSMSymbolicDataset:
    """Handler for GSM-Symbolic dataset from HuggingFace"""

    DSET_NAME = "apple/GSM-Symbolic"
    MAX_SETS = 50

    class Variant(Enum):
        main = auto()
        p1 = auto()
        p2 = auto()

    class Split(Enum):
        test = auto()

    def __init__(self, variant: Variant, split: Split = Split.test):
        """
        Load GSM-Symbolic dataset

        Args:
            variant: Dataset variant - 'main' (default) / 'p1' / 'p2'.
        """

        self._variant = self._check_type(variant, self.Variant)
        self._split = self._check_type(split, self.Split)

        logger.info(f"Loading GSM-Symbolic dataset (variant: {variant})...")
        self.dataset = load_dataset(self.DSET_NAME, self._variant.name, split=self._split.name)
        logger.info(f"Loaded {len(self.dataset)} examples")

    @staticmethod
    def _check_type(value: T, expected_type: type[T]) -> T:
        if not isinstance(value, expected_type):
            raise TypeError(f"Expected a {expected_type}, got {type(value)}: {value}")
        return value

    @property
    def variant_name(self) -> str:
        return self._variant.name

    @property
    def split_name(self) -> str:
        return self._split.name

    @property
    def path_friendly_dset_name(self) -> str:
        return make_name_path_friendly(self.DSET_NAME)

    @property
    def path_friendly_name(self) -> str:
        return make_name_path_friendly(f"{self.DSET_NAME}_{self._variant.name}_{self._split.name}")

    def get_subdataset_for_original_id(self, original_id: int) -> Dataset:
        """Get all instances of a specific question template"""

        return self.dataset.filter(
            lambda x: x == original_id,
            input_columns=["original_id"]
        )

    def get_unique_templates(self) -> list[int]:
        """Get list of unique template IDs"""

        return list(set(self.dataset['original_id']))

    def create_evaluation_sets(self, n_sets: int | None = None, n_per_set: int = None) -> list[list[dict]]:
        """
        Create evaluation sets (matching paper's methodology)
        Each set contains up to 100 examples (one per template)

        Returns:
            list of <num_instances> sets, each with 100 examples
        """

        templates = self.get_unique_templates()[:n_per_set]

        if n_sets is None:
            n_sets = self.MAX_SETS

        logger.info(f"Creating {n_sets} sets with {len(templates)} examples each")

        eval_sets = []

        for instance_idx in tqdm(range(n_sets), desc="set", position=0):
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
