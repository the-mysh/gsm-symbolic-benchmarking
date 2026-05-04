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

from gsm_benchmarker.results_analyser.multi_model import MultiModelResultsAnalyser
from gsm_benchmarker.results_analyser.plotting_utils import (plot_question_difficulty_matrix,
                                                             plot_question_difficulty_histogram, plot_number_counts)


logger = logging.getLogger(__name__)

try:
    from gsm_benchmarker.results_analyser.common import GLMMRunner, do_for_metrics
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

    def get_question_difficulty(self, model: str | None = None):
        """
        Compute a question difficulty score: proportion of models that got each question wrong ([0, 1]).

        If 'model' is provided, skip this model's results when calculating difficulty (leave-one-out model)
        to avoid circularity.
        """

        df = self._variants[self.BASELINE_VARIANT].full_data

        if model is not None:
            if model not in df.model.unique():
                logger.warning(f"'{model}' does not match any model; the results will be calculated for all models")
            else:
                df = df[df.model != model]

        difficulty = df.groupby('id')['correct'].mean().rename('difficulty')
        difficulty = 1 - difficulty  # invert results - highest difficulty gets highest score (lowest % solved)
        return difficulty

    def get_question_difficulty_per_model(self):
        difficulties = {}

        if len(self.models) == 1:
            difficulties[self.models[0]] = self.get_question_difficulty()  # array of zeros if only one model present
        else:
            for model in self.models:
                difficulties[model] = self.get_question_difficulty(model=model)

        difficulties['OVERALL'] = self.get_question_difficulty()

        difficulties_df = pd.DataFrame(difficulties).T
        return difficulties_df

    def plot_question_difficulty_per_model(self, **kwargs):
        difficulties = self.get_question_difficulty_per_model()
        return plot_question_difficulty_matrix(difficulties, **kwargs)

    def plot_question_difficulty_histogram(self, model: str | None = None,
                                           save_prefix: str | Path | None = None, **kwargs):
        difficulties = self.get_question_difficulty(model=model)

        if save_prefix is not None and model is not None:
            save_prefix = f"{save_prefix}_{model}"

        return plot_question_difficulty_histogram(difficulties, **kwargs, save_prefix=save_prefix)

    def _validate_models(self, models: list[str], variant: str):
        models_validated = []

        baseline_models = self.variants[self.BASELINE_VARIANT].models
        variant_models = self.variants[variant].models

        for model in models:
            if model not in baseline_models or model not in variant_models:
                logger.warning(f"No data for model {model}")
            else:
                models_validated.append(model)

        if not models_validated:
            raise ValueError(f"No data for any of the models: {', '.join(models)}")

        return models_validated

    @do_for_metrics
    def analyse_variant_effect(self, variant: str, metric: str, models: list[str] | None = None,
                               use_difficulty: bool = True):
        if models is not None:
            models = self._validate_models(models, variant)

        if GLMMRunner is None:
            raise RuntimeError("R not available")

        difficulty = self.get_question_difficulty_per_model() if use_difficulty else None
        glmm_runner = GLMMRunner(label='is_variant', question_difficulties=difficulty)
        data_df = glmm_runner.prep_df_with_bool_labels(
            metric=metric,
            ras={
                0: self.variants[self.BASELINE_VARIANT],
                1: self.variants[variant]
        })

        glmm_results_df = glmm_runner.run(df=data_df, models=models)

        # add plain accuracy drops
        glmm_results_df = glmm_results_df.join(self.get_mean_accuracy_summary(metric=metric))

        return glmm_results_df

    def _get_number_effect_glmm_data(self, variant: str, metric: str):
        df = self.variants[variant].full_data.reset_index()

        number_pattern = re.compile(r'\d*\.?\d+')
        def extract_sum_logs(text):
            matches = number_pattern.findall(text)
            if not matches:
                return np.nan # Handle the rare case where a prompt has no numbers
            numbers = (float(match) for match in matches)
            numbers = (number for number in numbers if not number%1)  # take integers only
            logs = ((np.log10(number) if number > 0 else 0) for number in numbers)
            return sum(logs)

        df_for_glmm = pd.DataFrame({
            'model': df.model,
            'id': df.id,
            'is_correct': df[metric].astype(int),
            'sum_logs': df.question.apply(extract_sum_logs)
        })

        return df_for_glmm

    @do_for_metrics
    def analyse_number_effect(self, variant: str, metric: str | None = None, models: list[str] | None = None):
        if models is not None:
            models = self._validate_models(models, variant)

        if GLMMRunner is None:
            raise RuntimeError("R not available")

        glmm_data = self._get_number_effect_glmm_data(variant=variant, metric=metric)

        glmm_runner = GLMMRunner(label="sum_logs")
        glmm_results_df = glmm_runner.run(df=glmm_data, models=models)
        glmm_results_df['odds_change'] = np.exp(glmm_results_df.estimate) - 1  # change in odds

        return glmm_results_df

    def get_mean_accuracy_summary(self, variant: str = 'main', metric: str | None = None) -> pd.DataFrame:
        acc_change = self.get_accuracy_summary(variant=variant, metric=metric)
        gb = ['model', 'metric'] if metric is None else ['model']
        return acc_change.groupby(gb).mean()

    def get_number_counts(self, model: str | None = None, bin_edges: list[int | float] | None = None):
        """Obtain counts of all numbers appearing in the questions present in data (for a single model)."""

        if bin_edges is None:
            bin_edges = [0, 1, 2, 3, 4, 5, 10, 20, 50, 100, 1000, float('inf')]

        bin_labels = ['fractions']
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
            raw_counts_dict[variant_name] = Counter(extracted_numbers)

            # put into bins, all fractions in a single separate bin
            integer_mask = np.isclose(extracted_numbers, np.round(extracted_numbers))
            integer_numbers = extracted_numbers[integer_mask]
            fraction_count = int((~integer_mask).sum())
            integer_binned = pd.cut(
                integer_numbers,
                bins=bin_edges,
                labels=bin_labels[1:],
                right=False,
                include_lowest=True,
            )
            integer_counts = integer_binned.value_counts().reindex(bin_labels[1:], fill_value=0)
            binned_counts = pd.Series([fraction_count] + integer_counts.values.tolist(), index=bin_labels, dtype=int)
            binned_counts_dict[variant_name] = binned_counts

        raw_counts_df = pd.DataFrame(raw_counts_dict).fillna(0).astype(int).sort_index()
        binned_counts_df = pd.DataFrame(binned_counts_dict).fillna(0).astype(int)

        return raw_counts_df, binned_counts_df

    def plot_number_counts(self, model: str | None = None, bin_edges: list[int | float] | None = None, **kwargs):
        raw_counts_df, binned_counts_df = self.get_number_counts(model=model, bin_edges=bin_edges)
        return plot_number_counts(raw_counts_df, binned_counts_df, **kwargs)
