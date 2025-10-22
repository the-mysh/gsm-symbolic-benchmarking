import pandas as pd
from pathlib import Path


class ModelResultsAnalyser:
    def __init__(self, data: pd.DataFrame):
        self._check_data(data)
        self._data = data
        self._data.index.names = ['set', 'question']  # TODO: move to where it's saved

    @staticmethod
    def _check_data(data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Expected a pandas.DataFrame, got {type(data)}: {data}")

        if 'correct' not in data.columns:
            raise ValueError("The provided dataframe is missing a 'correct' column")

        if not isinstance(data.index, pd.MultiIndex) or data.index.nlevels != 2:
            raise ValueError("The results dataframe must have a 2-level multiindex")

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @classmethod
    def from_file(cls, file_path: str | Path):
        return cls(pd.read_parquet(file_path))

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

