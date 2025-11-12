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
