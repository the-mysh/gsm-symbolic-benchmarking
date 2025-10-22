import os
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm


logger = logging.getLogger(__name__)


class ModelResultsAnalyser:
    def __init__(self, file_path: str | Path):
        self._file_path = Path(file_path)

        data = self.load_data(self._file_path)
        self._check_data(data)
        self._data = data
        self._data.index.names = ['set', 'question']  # TODO: move to where it's saved

    @staticmethod
    def load_data(file_path: str | Path) -> pd.DataFrame:
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
        self._dir_path = Path(dir_path)
        self._summary_data = self.load_summary_data(self._dir_path)

    @property
    def summary_data(self) -> pd.DataFrame:
        return self._summary_data

    @staticmethod
    def load_summary_data(dir_path: str | Path):
        dir_path = Path(dir_path).resolve()

        data_dict = {}

        logger.debug("Loading per-model results")
        for item_name in tqdm(os.listdir(dir_path), desc="Model"):
            item_path = dir_path / item_name
            if item_path.is_dir():
                logger.warning(f"The algorithm is not meant for non-flat directories; found subfolder '{item_name}'")
                continue
            model_results = ModelResultsAnalyser(item_path)
            s = model_results.get_total_accuracy_and_std()
            data_dict[''.join(item_name.split('.')[:-1])] = {'accuracy': s[0], 'std': s[1]}

        data_df = pd.DataFrame(data_dict)
        return data_df.T
