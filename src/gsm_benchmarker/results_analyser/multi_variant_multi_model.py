import os
import logging
import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from gsm_benchmarker.results_analyser.multi_model import MultiModelResultsAnalyser


logger = logging.getLogger(__name__)


class MultiVariantMultiModelResultsAnalyser:
    VARIANT_NAME_PATTERN = re.compile(r"[\w_-]+_(?P<variant>\w+)_test")

    def __init__(self, dir_path: str | Path):
        self._dir_path = Path(dir_path).resolve()
        self._summary_data, self._comparison_data = self._load_data(self._dir_path)

    @property
    def summary_data(self):
        return self._summary_data

    @property
    def comparison_data(self):
        return self._comparison_data

    @classmethod
    def match_variant_name(cls, name):
        match = cls.VARIANT_NAME_PATTERN.match(name)
        if not match:
            return name
        return match.group('variant')

    @classmethod
    def _load_data(cls, dir_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        summary_data_dict = {}
        comparative_data_dict = {}

        logger.debug("Loading per-model results")
        for item_name in tqdm(os.listdir(dir_path), desc="Model"):
            item_path = dir_path / item_name
            if not item_path.is_dir():
                continue
            multi_model_results = MultiModelResultsAnalyser(item_path, load_full_data=True)
            v = cls.match_variant_name(item_name)
            summary_data_dict[v] = multi_model_results.summary_data

            idx_frame = multi_model_results.full_data[['model', 'id', 'instance']]
            s = multi_model_results.full_data['correct']
            s.index = pd.MultiIndex.from_frame(idx_frame)
            comparative_data_dict[v] = s

        def concat(d: dict[str, pd.DataFrame | pd.Series]) -> pd.DataFrame:
            return pd.concat(d.values(), keys=d.keys(), axis=1)

        df_summary = concat(summary_data_dict)
        df_comparison = concat(comparative_data_dict).reset_index()
        return df_summary, df_comparison
