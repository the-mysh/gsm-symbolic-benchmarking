import logging
from datasets import load_dataset, Dataset
from enum import Enum, auto
from typing import TypeVar

from gsm_benchmarker.utils.path_ops import make_name_path_friendly
from gsm_benchmarker.answer_extractor import AnswerExtractor


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
        GSM8K = auto()

    class Split(Enum):
        test = auto()

    def __init__(self, variant: Variant, split: Split = Split.test):
        """
        Load GSM-Symbolic dataset

        Args:
            variant: Dataset variant - 'main' (default) / 'p1' / 'p2' / 'GSM8K' (constructed from 'main').
        """

        self._variant = self._check_type(variant, self.Variant)
        self._split = self._check_type(split, self.Split)
        self.dataset = self.load_dataset()
        self.answer_extractor = AnswerExtractor(code=False)

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
        return make_name_path_friendly(f"{self._variant.name}_{self._split.name}")

    def load_dataset(self):
        if self._variant is self.Variant.GSM8K:
            load_variant = self.Variant.main
            logger.debug(f"Loading GSM-Symbolic dataset in variant 'main'; original questions will be extracted from it")
        else:
            logger.debug(f"Loading GSM-Symbolic dataset (variant: {self._variant.name})...")
            load_variant = self._variant

        ds = load_dataset(self.DSET_NAME, load_variant.name, split=self._split.name)
        logger.debug(f"Loaded {len(ds)} examples")

        if self._variant is self.Variant.GSM8K:
            ds = self.get_subdataset_for_instance(min(ds['instance']), dset=ds)  # limit to 100 from 5000 examples

            def set_instance(d):
                d['instance'] = -1  # mark original questions as instance -1
                return d
            ds = ds.map(set_instance)

            ds = ds.remove_columns(['question', 'answer', 'canary'])
            ds = ds.rename_column('original_question', 'question')
            ds = ds.rename_column('original_answer', 'answer')

        else:
            ds = ds.remove_columns(['original_question', 'original_answer', 'canary'])

        # unify column order
        ds = ds.select_columns(['id', 'original_id', 'instance', 'question', 'answer'])

        # extract numerical results from answers
        ds = ds.map(lambda example: {'numerical_result': self.answer_extractor.extract_answer(example['answer'])[0]})

        logger.debug(f"After transformations: {len(ds)} examples")
        return ds

    def get_subdataset_for_instance(self, inst: int, dset: Dataset | None = None) -> Dataset:
        """Get all questions for a specific template instance"""

        if dset is None:
            dset = self.dataset

        dset_filtered = dset.filter(lambda x: x == inst, input_columns=["instance"])
        assert isinstance(dset_filtered, Dataset)  # type hinting
        return dset_filtered

    def create_evaluation_sets(self, n_sets: int | None = None, n_per_set: int = None) -> list[Dataset]:
        """
        Create evaluation sets (matching paper's methodology)
        Each set contains up to 100 examples (one per template)

        Returns:
            list of <num_instances> sets, each with 100 examples
        """

        if self._variant is self.Variant.GSM8K:
            if n_sets is not None and n_sets > 1:
                logger.warning(f"For variant {self._variant.GSM8K}, only one evaluation set can be created")
            n_sets = 1

        if n_sets is None:
            n_sets = self.MAX_SETS
            
        logger.info(f"Dataset variant {self._variant.name}: creating {n_sets} set(s) "
                    f"with {n_per_set or 'maximum available number of'} example(s) each")

        eval_sets = []
        all_instances = list(set(self.dataset['instance']))

        for instance_idx in range(n_sets):
            instance_dset = self.get_subdataset_for_instance(all_instances[instance_idx])
            if not len(instance_dset):
                logger.warning(f"Not enough instances for {n_sets} evaluation sets")
                break

            if n_per_set:
                instance_dset = instance_dset.select(range(n_per_set))

            eval_sets.append(instance_dset)

        return eval_sets
