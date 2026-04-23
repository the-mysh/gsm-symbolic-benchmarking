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
from gsm_benchmarker.results_analyser.plotting_utils import plot_question_success_rate_matrix, plot_question_difficulty_histogram
from gsm_benchmarker.results_analyser.common import GLMMRunner

logger = logging.getLogger(__name__)


class MultiVariantMultiModelResultsAnalyser:
    VARIANT_NAME_PATTERN = re.compile(r"(?P<variant>\w+)_test")
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

    def get_accuracy_change(self, variant: str, metric: str | None = None):
        self._check_variant(variant)

        baseline_accuracies = self._variants[self.BASELINE_VARIANT].get_accuracies_per_model_and_template_id(metric=metric)
        variant_accuracies = self._variants[variant].get_accuracies_per_model_and_template_id(metric=metric)

        acc_change = variant_accuracies - baseline_accuracies
        acc_change = acc_change.rename('accuracy_change')

        return acc_change

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
        return plot_question_success_rate_matrix(difficulties, **kwargs)

    def plot_question_difficulty_histogram(self, model: str | None = None,
                                           save_prefix: str | Path | None = None, **kwargs):
        difficulties = self.get_question_difficulty(model=model)

        if save_prefix is not None and model is not None:
            save_prefix = f"{save_prefix}_{model}"

        plot_question_difficulty_histogram(difficulties, **kwargs, save_prefix=save_prefix)

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

    def analyse_variant_effect(self, variant: str, models: list[str] | None = None, metric: str | None = None):
        if models is not None:
            models = self._validate_models(models, variant)

        glmm_runner = GLMMRunner(label='is_variant', question_difficulties=self.get_question_difficulty_per_model())
        glmm_results_df = glmm_runner.run(
            ras={
                0: self.variants[self.BASELINE_VARIANT],
                1: self.variants[variant]
            },
            models=models,
            metric=metric
        )

        # add plain accuracy drops
        if metric:
            acc_change = self.get_accuracy_change(variant, metric=metric).groupby('model')
        else:
            acc_change = self.get_accuracy_change(variant).groupby(['model', 'metric'])

        glmm_results_df['accuracy_change'] = acc_change.mean()

        return glmm_results_df
