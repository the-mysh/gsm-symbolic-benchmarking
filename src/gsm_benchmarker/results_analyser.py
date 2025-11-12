import os
import logging
import re
from functools import cached_property
import pandas as pd
from pathlib import Path
from tqdm import tqdm
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


class MultiModelResultsAnalyser:
    def __init__(self, dir_path: str | Path, load_full_data: bool = False):
        self._dir_path = Path(dir_path).resolve()

        summary_data_dict, full_data_dict = self._load_data(self._dir_path, load_full_data=load_full_data)
        self._summary_data = self._make_summary_df(summary_data_dict)
        self._full_data = self._make_full_df(full_data_dict) if full_data_dict else None

    @cached_property
    def full_data(self) -> pd.DataFrame:
        if self._full_data is None:
            self._full_data = self._load_full_data()
        return self._full_data

    @property
    def summary_data(self) -> pd.DataFrame:
        return self._summary_data

    @staticmethod
    def _load_data(dir_path: Path, load_full_data: bool = False):
        full_data_dict = {}
        summary_data_dict = {}

        logger.debug("Loading per-model results")
        for item_name in tqdm(os.listdir(dir_path), desc="Model"):
            item_path = dir_path / item_name
            if item_path.is_dir():
                logger.warning(f"The algorithm is not meant for non-flat directories; found subfolder '{item_name}'")
                continue
            model_results = ModelResultsAnalyser(item_path)
            model_name = ''.join(item_name.split('.')[:-1])

            if load_full_data:
                full_data_dict[model_name] = model_results.data
            s = model_results.get_total_accuracy_and_std()
            summary_data_dict[model_name] = {'accuracy': s[0], 'std': s[1]}

        return summary_data_dict, full_data_dict

    @staticmethod
    def _make_summary_df(summary_data_dict):
        data_df = pd.DataFrame(summary_data_dict)
        return data_df.T

    @staticmethod
    def _make_full_df(full_data_dict):
        df = pd.concat(full_data_dict.values(), keys=full_data_dict.keys(), names=['model', 'old_index'])
        df = df.reset_index().drop('old_index', axis=1)
        return df

    def _load_full_data(self):
        _, data_dict = self._load_data(dir_path=self._dir_path, load_full_data=True)
        return self._make_full_df(data_dict)

    @property
    def models(self) -> list[str]:
        return self.full_data.model.unique().tolist()

    @property
    def instances(self) -> list[int]:
        return self.full_data.instance.unique().tolist()

    @property
    def ids(self) -> list[int]:
        return self.full_data.id.unique().tolist()

    def filter(self, **pairs: Any) -> pd.DataFrame:
        df = self.full_data
        for (column, value) in pairs.items():
            df = df[df[column] == value]
        return df

    def get_example(self, id: int, instance: int, model: str) -> dict[str, Any] | None:
        df = self.filter(id=id, instance=instance, model=model)

        if not len(df):
            if model not in self.models:
                raise ValueError(f"Model {model} does not exist in data")
            if id not in self.ids:
                raise ValueError(f"Id {id} does not exist in data")
            if instance not in self.instances:
                raise ValueError(f"Instance {instance} does not exist in data")

            # each exists, just not the combo
            logger.warning(f"No example with template id {id}, instance number {instance},"
                           f"and model {model} found")
            return None

        if len(df) > 1:
            raise RuntimeError(f"Multiple examples with the same template id {id}, "
                               f"instance number {instance}, and model {model} found")

        return df.to_dict(orient='index')[df.index[0]]


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
            if not item_path.is_dir():
                continue
            multi_model_results = MultiModelResultsAnalyser(item_path)
            data_dict[cls.match_variant_name(item_name)] = multi_model_results.summary_data

        df = pd.concat(data_dict.values(), keys=data_dict.keys(), axis=1)
        return df