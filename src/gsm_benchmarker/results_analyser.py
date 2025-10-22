import os
import logging
import re
from functools import cached_property
import pandas as pd
from pathlib import Path
from tqdm import tqdm


logger = logging.getLogger(__name__)


class ModelResultsAnalyser:
    def __init__(self, file_path: str | Path):
        self._file_path = Path(file_path)

        data = self._load_data(self._file_path)
        self._check_data(data)
        self._data = data
        self._data.index.names = ['set_number', 'question_number']  # TODO: move to where it's saved

    @staticmethod
    def _load_data(file_path: str | Path) -> pd.DataFrame:
        return pd.read_parquet(file_path)

    @staticmethod
    def _check_data(data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Expected a pandas.DataFrame, got {type(data)}: {data}")

        if 'correct' not in data.columns:
            raise ValueError("The results dataframe is missing a 'correct' column")

        if not isinstance(data.index, pd.MultiIndex) or data.index.nlevels != 2:
            raise ValueError("The results dataframe must have a 2-level multiindex")

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def correctness(self) -> pd.Series:
        return self._data['correct']

    def get_accuracy_per_set(self) -> pd.Series:
        return self.correctness.groupby(level=0).mean()

    def get_accuracy_per_question(self) -> pd.Series:
        return self.correctness.groupby(level=1).mean()

    def get_total_accuracy_and_std(self) -> tuple[float, float | None]:
        """Compute mean of accuracies per set and the corresponding standard deviation (if more than 1 set)."""

        accuracies = self.get_accuracy_per_set()
        mean_acc = float(accuracies.mean())
        std_acc = float(accuracies.std()) if len(accuracies) > 1 else None
        return mean_acc, std_acc


class MultiModelResultsAnalyser:
    def __init__(self, dir_path: str | Path):
        self._dir_path = Path(dir_path).resolve()
        self._summary_data = self._load_summary_data(self._dir_path)
        self._full_data = None

    @cached_property
    def full_data(self) -> pd.DataFrame:
        if self._full_data is None:
            self._full_data = self._load_full_data()
        return self._full_data

    @property
    def summary_data(self) -> pd.DataFrame:
        return self._summary_data

    @staticmethod
    def _load_data(dir_path: Path, full: bool = False):
        data_dict = {}

        logger.debug("Loading per-model results")
        for item_name in tqdm(os.listdir(dir_path), desc="Model"):
            item_path = dir_path / item_name
            if item_path.is_dir():
                logger.warning(f"The algorithm is not meant for non-flat directories; found subfolder '{item_name}'")
                continue
            model_results = ModelResultsAnalyser(item_path)
            model_name = ''.join(item_name.split('.')[:-1])
            if full:
                data_dict[model_name] = model_results.data
            else:
                s = model_results.get_total_accuracy_and_std()
                data_dict[model_name] = {'accuracy': s[0], 'std': s[1]}

        return data_dict

    @staticmethod
    def _load_summary_data(dir_path):
        data_dict = MultiModelResultsAnalyser._load_data(dir_path)
        data_df = pd.DataFrame(data_dict)
        return data_df.T

    def _load_full_data(self):
        data_dict = self._load_data(dir_path=self._dir_path, full=True)
        df = pd.concat((v.reset_index() for v in data_dict.values()), keys=data_dict.keys())

        return df


class MultiVariantMultiModelResultsAnalyser:
    VARIANT_NAME_PATTERN = re.compile(r"[\w_-]+_(?P<variant>\w+)_test")

    def __init__(self, dir_path: str | Path):
        self._dir_path = Path(dir_path).resolve()
        self._summary_data = self._load_summary_data(self._dir_path)

    @property
    def summary_data(self):
        return self._summary_data

    @classmethod
    def match_variant_name(cls, name):
        match = cls.VARIANT_NAME_PATTERN.match(name)
        if not match:
            return name
        return match.group('variant')

    @classmethod
    def _load_summary_data(cls, dir_path: Path) -> pd.DataFrame:
        data_dict = {}

        logger.debug("Loading per-model results")
        for item_name in tqdm(os.listdir(dir_path), desc="Model"):
            item_path = dir_path / item_name
            multi_model_results = MultiModelResultsAnalyser(item_path)
            data_dict[cls.match_variant_name(item_name)] = multi_model_results.summary_data

        df = pd.concat(data_dict.values(), keys=data_dict.keys(), axis=1)
        return df