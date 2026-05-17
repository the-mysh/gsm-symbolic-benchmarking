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
from collections import Counter

from gsm_benchmarker.results_analyser.common import do_for_metrics
from gsm_benchmarker.results_analyser.multi_model import MultiModelResultsAnalyser
from gsm_benchmarker.results_analyser.plotting_utils import plot_number_counts


logger = logging.getLogger(__name__)

try:
    from gsm_benchmarker.results_analyser.common import GLMMRunner
except (ValueError, ImportError) as exc:
    logger.warning("R not configured, some functions will not be available")
    logger.warning(exc)
    GLMMRunner = None


class MultiVariantMultiModelResultsAnalyser:
    VARIANT_NAME_PATTERN = re.compile(r"(?P<variant>\w+)_test")
    NUMBER_PATTERN = re.compile(r'\d+(?:\.\d+)?')
    BASELINE_VARIANT = 'GSM8K'

    def __init__(self, dir_path: str | Path):
        self._dir_path = Path(dir_path).resolve()
        self._summary_data, self._variants = self._load_data(self._dir_path)

    @property
    def summary_data(self):
        return self._summary_data

    @property
    def variants(self):
        return self._variants

    @property
    def models(self) -> list[str]:
        return self._summary_data.index.tolist()

    @classmethod
    def match_variant_name(cls, name):
        match = cls.VARIANT_NAME_PATTERN.match(name)
        if not match:
            return name
        return match.group('variant')

    @classmethod
    def _load_data(cls, dir_path: Path) -> tuple[pd.DataFrame, dict[str, MultiModelResultsAnalyser]]:
        summary_data_dict = {}
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

        def concat(d: dict[str, pd.DataFrame | pd.Series]) -> pd.DataFrame:
            return pd.concat(d.values(), keys=d.keys(), axis=1)

        df_summary = concat(summary_data_dict)

        return df_summary, variants

    def _check_variant(self, variant: str):
        if variant not in self._variants:
            raise ValueError(f"No data for variant '{variant}'")

        if variant == self.BASELINE_VARIANT:
            raise ValueError(f"{self.BASELINE_VARIANT} is the baseline variant "
                             f"- choose a different variant to compare it to")

    def get_accuracy_summary(self, variant: str, metric: str | None = None):
        self._check_variant(variant)

        baseline_accuracies = self._variants[self.BASELINE_VARIANT].get_accuracies_per_model_and_template_id(metric=metric)
        variant_accuracies = self._variants[variant].get_accuracies_per_model_and_template_id(metric=metric)

        acc_data = pd.DataFrame({
            self.BASELINE_VARIANT + '_acc': baseline_accuracies,
            variant + '_acc': variant_accuracies,
            'acc_diff': variant_accuracies - baseline_accuracies
        })

        return acc_data

    def get_baseline_comparison_df(self, variant: str, model: str | None = None):
        self._check_variant(variant)

        baseline_subset = self._variants[self.BASELINE_VARIANT].full_data[['model', 'id', 'correct', 'result_class']]
        baseline_subset = baseline_subset.rename(
            columns={'correct': 'baseline_correct', 'result_class': 'baseline_result_class'})

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


    def _validate_models(self, models: list[str] | None, variant: str):
        baseline_models = self.variants[self.BASELINE_VARIANT].models
        variant_models = self.variants[variant].models

        if models is None:
            models = list(set(baseline_models + variant_models))

        models_validated = []

        for model in models:
            if model not in baseline_models:
                logger.warning(f"No baseline data for model {model}")
            elif model not in variant_models:
                logger.warning(f"No variant data for model {model}")
            else:
                models_validated.append(model)

        if not models_validated:
            raise ValueError(f"No data for any of the models: {', '.join(models)}")

        return models_validated

    @do_for_metrics
    def analyse_variant_effect(self, variant: str, metric: str, models: list[str] | None = None):
        models = self._validate_models(models, variant)

        if GLMMRunner is None:
            raise RuntimeError("R not available")

        glmm_runner = GLMMRunner('is_variant')
        data_df = glmm_runner.prep_df_with_bool_labels(
            metric=metric,
            ras={
                0: self.variants[self.BASELINE_VARIANT],
                1: self.variants[variant]
        })

        glmm_results_df = glmm_runner.run(df=data_df, models=models, simplify=True)

        # add plain accuracy drops
        glmm_results_df = glmm_results_df.join(self.get_mean_accuracy_summary(metric=metric))

        return glmm_results_df

    def _get_number_effect_glmm_data(self, variant: str, metric: str):
        number_pattern = re.compile(r'\d*\.?\d+')

        def extract_sum_logs(text):
            matches = number_pattern.findall(text)
            if not matches:
                return np.nan  # Handle the rare case where a prompt has no numbers
            numbers = (float(match) for match in matches)
            numbers = (number for number in numbers if not number % 1)  # take integers only
            logs = ((np.log10(number) if number > 0 else 0) for number in numbers)
            return sum(logs)

        def _prep(res, variant_label: bool):
            data = res.full_data[['model', 'id', metric, 'question']][:].reset_index(drop=True)
            data['is_variant'] = [int(variant_label)] * len(data)
            data['is_correct'] = data[metric]
            data['id'] = data['id']
            data['sum_logs'] = data.question.apply(extract_sum_logs)

            data = data.drop(metric, axis=1).drop('question', axis=1)
            return data

        baseline_df = _prep(self.variants[self.BASELINE_VARIANT], False)
        variant_df = _prep(self.variants[variant], True)

        df = pd.concat([baseline_df, variant_df]).reset_index(drop=True)
        df['sum_logs_c'] = df['sum_logs'] - df['sum_logs'].mean()
        df = df.drop('sum_logs', axis=1)
        return df

    @do_for_metrics
    def analyse_number_effect(self, variant: str, metric: str, models: list[str] | None = None):
        if models is not None:
            models = self._validate_models(models, variant)

        if GLMMRunner is None:
            raise RuntimeError("R not available")

        glmm_data = self._get_number_effect_glmm_data(variant=variant, metric=metric)

        glmm_runner = GLMMRunner("sum_logs_c")
        glmm_results_df = glmm_runner.run(df=glmm_data, models=models, simplify=True)
        odds_ratio = np.exp(glmm_results_df.estimate)
        glmm_results_df['odds_ratio'] = odds_ratio
        glmm_results_df['odds_change'] = odds_ratio - 1  # change in odds

        return glmm_results_df

    def get_mean_accuracy_summary(self, variant: str = 'main', metric: str | None = None) -> pd.DataFrame:
        acc_change = self.get_accuracy_summary(variant=variant, metric=metric)
        gb = ['model', 'metric'] if metric is None else ['model']
        return acc_change.groupby(gb).mean()

    def get_number_counts(self, model: str | None = None, bin_edges: list[int | float] | None = None):
        """Obtain counts of all numbers appearing in the questions present in data (for a single model)."""

        if bin_edges is None:
            bin_edges = [0, 1, 2, 3, 4, 5, 10, 20, 50, 100, 1000, float('inf')]

        bin_labels = []
        for start, end in zip(bin_edges[:-1], bin_edges[1:]):
            if np.isinf(end):
                label = f"{start}+"
            elif (end - start) == 1:
                label = f"{start}"
            else:
                label = f'{start}-{end}'
            bin_labels.append(label)

        binned_counts_dict = {}
        raw_counts_dict = {}
        for variant_name in self.variants:
            variant_df = self.variants[variant_name].full_data
            model_df = variant_df.loc[variant_df['model'] == (model or self.models[0]), ['question']].copy()
            extracted_numbers = model_df['question'].str.findall(self.NUMBER_PATTERN).explode().dropna().astype(float)
            extracted_numbers = extracted_numbers[~(extracted_numbers % 1).astype(bool)]  # limit to integers
            raw_counts_dict[variant_name] = Counter(extracted_numbers)

            # put into bins, all fractions in a single separate bin
            binned = pd.cut(extracted_numbers, bins=bin_edges, labels=bin_labels, right=False, include_lowest=True)
            number_counts = binned.value_counts().reindex(bin_labels, fill_value=0)
            binned_counts_dict[variant_name] = number_counts

        raw_counts_df = pd.DataFrame(raw_counts_dict).fillna(0).astype(int).sort_index()
        binned_counts_df = pd.DataFrame(binned_counts_dict).fillna(0).astype(int)

        return raw_counts_df, binned_counts_df

    def plot_number_counts(self, model: str | None = None, bin_edges: list[int | float] | None = None, **kwargs):
        raw_counts_df, binned_counts_df = self.get_number_counts(model=model, bin_edges=bin_edges)
        return plot_number_counts(raw_counts_df, binned_counts_df, **kwargs)
