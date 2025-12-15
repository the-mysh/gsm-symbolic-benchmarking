import os
import logging
import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from gsm_benchmarker.results_analyser.multi_model import MultiModelResultsAnalyser


logger = logging.getLogger(__name__)


class MultiVariantMultiModelResultsAnalyser:
    VARIANT_NAME_PATTERN = re.compile(r"(?P<variant>\w+)_test")
    BASELINE_VARIANT = 'GSM8K'

    def __init__(self, dir_path: str | Path):
        self._dir_path = Path(dir_path).resolve()
        self._summary_data, self._comparison_data, self._variants = self._load_data(self._dir_path)

    @property
    def summary_data(self):
        return self._summary_data

    @property
    def comparison_data(self):
        return self._comparison_data

    @property
    def variants(self):
        return self._variants

    @classmethod
    def match_variant_name(cls, name):
        match = cls.VARIANT_NAME_PATTERN.match(name)
        if not match:
            return name
        return match.group('variant')

    @classmethod
    def _load_data(cls, dir_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, MultiModelResultsAnalyser]]:
        summary_data_dict = {}
        comparative_data_dict = {}
        variants = {}

        logger.debug("Loading per-model results")
        for item_name in tqdm(os.listdir(dir_path), desc="Model"):
            item_path = dir_path / item_name
            if not item_path.is_dir():
                continue
            multi_model_results = MultiModelResultsAnalyser(item_path, load_full_data=True)
            v = cls.match_variant_name(item_name)
            summary_data_dict[v] = multi_model_results.summary_data
            variants[v] = multi_model_results

            idx_frame = multi_model_results.full_data[['model', 'id', 'instance']]
            s = multi_model_results.full_data['correct']
            s.index = pd.MultiIndex.from_frame(idx_frame)
            comparative_data_dict[v] = s

        def concat(d: dict[str, pd.DataFrame | pd.Series]) -> pd.DataFrame:
            return pd.concat(d.values(), keys=d.keys(), axis=1)

        df_summary = concat(summary_data_dict)

        comparative_data_dict = cls._fix_comparison_data(comparative_data_dict)
        df_comparison = concat(comparative_data_dict).reset_index()
        return df_summary, df_comparison, variants

    @staticmethod
    def _fix_comparison_data(data: dict) -> dict:
        gsm8k_keys = [k for k in data.keys() if 'gsm8k' in k.lower()]
        if not gsm8k_keys:
            return data
        if len(gsm8k_keys) > 1:
            logger.warning("Multiple GSM8K columns detected")
            return data

        k = gsm8k_keys[0]
        gsm = data.pop(k)
        gsm = gsm.reset_index().drop('instance', axis=1)

        all_instances = []
        for dset in data.values():
            all_instances.extend(dset.reset_index().instance.unique())
        df_instances = pd.DataFrame({'instance': list(set(all_instances))})

        gsm_new = gsm.merge(df_instances, how='cross')
        gsm_new = gsm_new.set_index(['model', 'id', 'instance'])[['correct']]

        data[k] = gsm_new

        return data

    def get_baseline_comparison_df(self, variant: str):
        if variant not in self._variants:
            raise ValueError(f"No data for variant '{variant}'")

        if variant == self.BASELINE_VARIANT:
            raise ValueError(f"{self.BASELINE_VARIANT} is the baseline variant "
                             f"- choose a different variant to compare it to")

        baseline_subset = self._variants[self.BASELINE_VARIANT].full_data[['model', 'id', 'correct', 'result_class']]
        baseline_subset = baseline_subset.rename(columns={'correct': 'baseline_correct', 'result_class': 'baseline_result_class'})

        variant_subset = self._variants[variant].full_data[['model', 'id', 'instance', 'correct', 'result_class']]

        merged = variant_subset.merge(baseline_subset, on=['model', 'id'], how='left')

        merged['diff_correct'] = merged['correct'].astype(int) - merged['baseline_correct'].astype(int)

        return merged

    def plot_baseline_transition_matrices(self, variant: str, subtitle: str | None = None):
        df = self.get_baseline_comparison_df(variant)

        correct_order = [True, False]
        correct_transition_matrix = pd.crosstab(
            df['baseline_correct'],
            df['correct'],
            normalize='all'
        ).reindex(
            index=correct_order,
            columns=correct_order,
            fill_value=0
        )

        rc_order = ['CORRECT', 'BABBLING', 'INCORRECT', 'FAILED']
        result_class_transition_matrix = pd.crosstab(
            df['baseline_result_class'],
            df['result_class'],
            normalize='all'
        ).reindex(
            index=rc_order,
            columns=rc_order,
            fill_value=0
        )

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        for i, (title, matrix) in enumerate((
                ('correctness', correct_transition_matrix),
                ('result class', result_class_transition_matrix)
        )):
            ax = axes[i]
            sns.heatmap(matrix, annot=True, fmt=".1%", cmap="YlGnBu", ax=ax)
            ax.set_title(title.capitalize())
            ax.set_ylabel("Baseline (from)")
            ax.set_xlabel("Template variations (to)")
            ax.set_aspect('equal')

        t = "Transition of results: baseline (GSM8K) -> template variations (GSM Symbolic)"
        if subtitle:
            t += ("\n" + subtitle)
        fig.suptitle(t)
