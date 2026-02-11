import os
import logging
import re
import numpy as np
from scipy import stats
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

        logger.debug("Loading results")
        for item_name in tqdm(os.listdir(dir_path)):
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

    def _check_variant(self, variant: str):
        if variant not in self._variants:
            raise ValueError(f"No data for variant '{variant}'")

        if variant == self.BASELINE_VARIANT:
            raise ValueError(f"{self.BASELINE_VARIANT} is the baseline variant "
                             f"- choose a different variant to compare it to")

    def get_accuracy_drop_df(self, variant: str):
        self._check_variant(variant)

        baseline_accuracies = self._variants[self.BASELINE_VARIANT].get_accuracies_per_model_and_template_id()
        variant_accuracies = self._variants[variant].get_accuracies_per_model_and_template_id()

        drop = baseline_accuracies - variant_accuracies
        drop = drop.rename(columns={'correct': 'accuracy_drop', 'correct_strict': 'strict_accuracy_drop'})

        return drop

    def get_baseline_comparison_df(self, variant: str, model: str | None = None):
        self._check_variant(variant)

        baseline_subset = self._variants[self.BASELINE_VARIANT].full_data[['model', 'id', 'correct', 'result_class']]
        baseline_subset = baseline_subset.rename(columns={'correct': 'baseline_correct', 'result_class': 'baseline_result_class'})

        variant_subset = self._variants[variant].full_data[['model', 'id', 'instance', 'correct', 'result_class']]

        if model is not None:
            baseline_subset = baseline_subset[baseline_subset.model == model]
            variant_subset = variant_subset[variant_subset.model == model]

        merged = variant_subset.merge(baseline_subset, on=['model', 'id'], how='left')

        merged['diff_correct'] = merged['correct'].astype(int) - merged['baseline_correct'].astype(int)

        return merged

    def run_gap_analysis(self, metric: str = 'correct', variant: str = 'main'):
        """
        Run one-tailed Wilcoxon signed-rank test (per model) to check whether accuracy drop is significant.
        """

        df_gsm8k = self._variants[self.BASELINE_VARIANT].full_data
        df_variant = self._variants[variant].full_data

        results = []

        for model in df_gsm8k.model.unique():

            # filter by model, aggregate by template id
            scores_gsm8k = df_gsm8k[df_gsm8k.model == model].groupby('id')[metric].mean()
            scores_variant = df_variant[df_variant.model == model].groupby('id')[metric].mean()

            # pair the corresponding attempts by template id
            # inner join - only compare ids present in both sets
            paired = pd.concat([scores_gsm8k, scores_variant], axis=1, join='inner')
            paired.columns = ['gsm8k', 'variant']

            # 4. Calculate Stats
            mean_gsm8k = paired['gsm8k'].mean()
            mean_variant = paired['variant'].mean()
            gap = mean_gsm8k - mean_variant

            # one-sided Wilcoxon test
            # H0: median(gsm8k - variant) <= 0
            # H1: median(gsm8k - variant) > 0  (the drop is real)
            if gap:
                stat, p_value = stats.wilcoxon(
                    x=paired['gsm8k'],
                    y=paired['variant'],
                    alternative='greater'
                )
            else:
                p_value = 1.0
                stat = np.nan

            results.append({'model': model, 'p_value': p_value, 'gap': gap, 'stat': stat})

        return pd.DataFrame(results)

    @staticmethod
    def _make_transition_matrix(data, order, column, margins_name='total'):
        order = order + [margins_name]

        counts_matrix = pd.crosstab(
            data[f'baseline_{column}'],
            data[column],
            margins=True,
            margins_name=margins_name,
        ).reindex(
            index=order,
            columns=order,
            fill_value=0
        )

        percentages_matrix = pd.crosstab(
            data[f'baseline_{column}'],
            data[column],
            normalize='all',
            margins=True,
            margins_name=margins_name,
        ).reindex(
            index=order,
            columns=order,
            fill_value=0
        )

        labels_matrix = (
                counts_matrix.astype(str) + "\n" +
                percentages_matrix.map(lambda x: f"({x:.1%})")
        )

        return percentages_matrix, labels_matrix

    def plot_baseline_transition_matrices(self, variant: str, subtitle: str | None = None, model: str | None = None):
        df = self.get_baseline_comparison_df(variant, model=model)

        correct_tm, correct_labels = self._make_transition_matrix(df, [True, False], 'correct')

        rc_tm, rc_labels = self._make_transition_matrix(
            df, ['CORRECT', 'BABBLING', 'INCORRECT', 'FAILED'], 'result_class')


        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        for i, (title, matrix, labels) in enumerate((
                ('numerical correctness', correct_tm, correct_labels),
                ('result class', rc_tm, rc_labels)
        )):
            ax = axes[i]
            sns.heatmap(matrix, annot=labels, fmt="", cmap="YlGnBu", ax=ax)
            ax.set_title(title.capitalize())
            ax.set_ylabel("GSM8K questions")
            ax.set_xlabel("GSM-Symbolic template variations")
            ax.set_aspect('equal')
            ax.xaxis.tick_top()                 # Move the ticks to the top
            ax.xaxis.set_label_position('top')  # Move the axis label to the top

            for func in (ax.axhline, ax.axvline):
                func(len(matrix) - 1, c='white', zorder=3, lw=8)

        t = "Transition of results: original GSM8K questions -> GSM-Symbolic template variations"
        if subtitle:
            t += ("\n" + subtitle)
        if model is not None:
            t += ((", " if subtitle else "\n") + model.replace("_", " "))
        fig.suptitle(t)

        fig.subplots_adjust(top=0.8, bottom=0.05)

        return fig