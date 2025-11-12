import logging
import pandas as pd
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


class ModelResultsAnalyser:
    def __init__(self, file_path: str | Path):
        self._file_path = Path(file_path)

        data = self._load_data(self._file_path)
        data = self._check_data(data)
        self._data = data

    @staticmethod
    def _load_data(file_path: str | Path) -> pd.DataFrame:
        return pd.read_parquet(file_path)

    @staticmethod
    def _check_data(data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Expected a pandas.DataFrame, got {type(data)}: {data}")

        if isinstance(data.index, pd.MultiIndex):
            # old version of results  # TODO: remove
            if data.index.nlevels != 2:
                raise ValueError("If multi-indexed, the results dataframe must have a 2 levels")
            data = data.reset_index().drop('set_number', axis=1).drop('question_number', axis=1)  # repeated

        for c in ('id', 'original_id', 'instance', 'correct'):
            if c not in data.columns:
                raise ValueError(f"The results dataframe is missing a '{c}' column")

        return data

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def correctness(self) -> pd.Series:
        return self._data['correct']

    def get_accuracy_per_instance(self) -> pd.Series:
        return self._data.groupby('instance').correct.mean()

    def get_accuracy_per_template_id(self) -> pd.Series:
        return self._data.groupby('id').correct.mean()

    def get_total_accuracy_and_std(self) -> tuple[float, float | None]:
        """Compute mean of accuracies per set and the corresponding standard deviation (if more than 1 set)."""

        accuracies = self.get_accuracy_per_instance()
        mean_acc = float(accuracies.mean())
        std_acc = float(accuracies.std()) if len(accuracies) > 1 else None
        return mean_acc, std_acc

    @property
    def instances(self) -> list[int]:
        return self.data.instance.unique().tolist()

    @property
    def ids(self) -> list[int]:
        return self.data.id.unique().tolist()

    def filter(self, **pairs: Any) -> pd.DataFrame:
        df = self._data
        for (column, value) in pairs.items():
            df = df[df[column] == value]
        return df

    def get_example(self, id: int, instance: int) -> dict[str, Any] | None:
        df = self.filter(id=id, instance=instance)

        if not len(df):
            if id not in self.ids:
                raise ValueError(f"Id {id} does not exist in data")
            if instance not in self.instances:
                raise ValueError(f"Instance {instance} does not exist in data")

            # both id and instance exist in data, but no example for this combination
            logger.warning(f"No example with template id {id} and instance number {instance} found")
            return None

        if len(df) > 1:
            raise RuntimeError(f"Multiple examples with the same template id {id} "
                               f"and instance number {instance} found")

        return df.to_dict(orient='index')[df.index[0]]
