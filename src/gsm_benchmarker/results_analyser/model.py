"""Single-model results loading and convenience accessors.

This module provides a thin wrapper around a parquet results file, offering
convenience methods to compute per-instance and per-template accuracies and
to extract example rows. The wrapper also augments the raw results with a
few derived columns (e.g. 'babbling', 'result_class').
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Any

from gsm_benchmarker.benchmark.answer_extractor import AnswerExtractor


logger = logging.getLogger(__name__)


class ModelResultsAnalyser:
    """Load and analyse a single model's results file.

    Parameters
    ----------
    file_path:
        Path to a parquet file containing per-example results.
    """

    def __init__(self, file_path: str | Path):
        self._file_path = Path(file_path)

        data = self._load_data(self._file_path)
        data = self._check_data(data)
        data = self._enhance_data(data)
        self._data = data

    @staticmethod
    def _load_data(file_path: str | Path) -> pd.DataFrame:
        return pd.read_parquet(file_path)

    @staticmethod
    def _enhance_data(data):
        """Insert additional derived columns into the results DataFrame.

        Adds boolean column 'babbling', stricter correctness 'correct_strict'
        (which discounts babbling answers) and a categorical 'result_class'
        labeling each example as CORRECT, BABBLING, INCORRECT, or FAILED.
        """

        # add 'babbling' column
        def b(s: str) -> bool:
            return any(bt in s for bt in AnswerExtractor.BABBLER_TOKENS)

        data['babbling'] = data.full_response.apply(b)
        data['correct_strict'] = data.correct.to_numpy() * ~data.babbling.to_numpy()

        # add 'result class' column - whether the answer was correct / correct+babbling / incorrect / failed to answer
        nan_idx = data.predicted_numerical_result.isna().to_numpy()
        babbling_idx = data.babbling.to_numpy()
        correct_answer_idx = data.correct.to_numpy()

        overall_correct = correct_answer_idx & ~nan_idx
        overall_incorrect = ~overall_correct

        strict_correct = overall_correct & ~babbling_idx
        babbling_correct = overall_correct & babbling_idx

        strict_incorrect = overall_incorrect & ~nan_idx
        failed_incorrect = overall_incorrect & nan_idx

        if strict_correct.sum() + babbling_correct.sum() + strict_incorrect.sum() + failed_incorrect.sum() != len(data):
            logger.error("Can't assign result class")
        else:
            data.loc[failed_incorrect, 'result_class'] = "FAILED"
            data.loc[strict_correct, 'result_class'] = "CORRECT"
            data.loc[babbling_correct, 'result_class'] = "BABBLING"
            data.loc[strict_incorrect, 'result_class'] = "INCORRECT"

        return data

    @staticmethod
    def _check_data(data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Expected a pandas.DataFrame, got {type(data)}: {data}")

        if isinstance(data.index, pd.MultiIndex):
            # old version of results  # TODO: remove
            if data.index.nlevels != 2:
                raise ValueError("If multi-indexed, the results dataframe must have a 2 levels")
            data = data.reset_index().drop('set_number', axis=1).drop('question_number', axis=1)  # repeated

        for c in ('id', 'original_id', 'instance', 'correct', 'predicted_numerical_result'):
            if c not in data.columns:
                raise ValueError(f"The results dataframe is missing a '{c}' column")

        return data

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    def _get_accuracy_per(self, col: str, strict: bool = False):
        if strict:
            ok = self._data.correct & ~self._data.babbling
        else:
            ok = self._data.correct

        data = pd.DataFrame({col: self._data[col], 'ok': ok})

        return data.groupby(col).ok.mean() * 100

    def get_accuracy_per_instance(self, strict: bool = False) -> pd.Series:
        """Return accuracy per instance number (in percent).

        If strict is True, babbling answers are treated as incorrect.
        """

        return self._get_accuracy_per('instance', strict=strict)

    def get_accuracy_per_template_id(self, strict: bool = False) -> pd.Series:
        return self._get_accuracy_per('id', strict=strict)

    def get_total_accuracy_and_std(self, strict: bool = False) -> tuple[float, float | None]:
        """Compute mean accuracy and standard deviation across instances.

        Returns (mean, std). If only a single instance is present, std is
        returned as None.
        """

        accuracies = self.get_accuracy_per_instance(strict=strict)
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
        """Filter the results DataFrame by equality of given column/value pairs.

        Example: filter(model='gemma-2-2b', id=12)
        """

        df = self._data
        for (column, value) in pairs.items():
            df = df[df[column] == value]
        return df

    def get_example(self, id: int, instance: int) -> dict[str, Any] | None:
        """Return a single example (row) for a given template id and instance.

        Raises informative errors when the requested id or instance is not
        present. Returns the row as a dictionary when found.
        """

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
