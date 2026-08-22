"""Analysis helpers for collections of dataset variants across models.

This module supports comparing a baseline dataset variant (GSM8K) with one or
more GSM-Symbolic variants. It provides statistical tests (Wilcoxon,
GLMM-based) and utilities for plotting transitions and number-counts that
appear in questions.
"""

import os
import logging
import re
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from gsm_benchmarker.results_analyser.utils import do_for_metrics
from gsm_benchmarker.results_analyser.multi_model import MultiModelResultsAnalyser
from gsm_benchmarker.results_analyser.plotting_utils import plot_number_counts
from gsm_benchmarker.results_analyser.bootstrap_result import BootstrapResult


logger = logging.getLogger(__name__)

try:
    from gsm_benchmarker.results_analyser.glmm import GLMMRunner
except (ValueError, ImportError) as exc:
    logger.warning("R not configured, some functions will not be available")
    logger.warning(exc)
    GLMMRunner = None


VARIANT_LABELS = {
    'main': 'GSM-Variants',
    'GSM8K': 'GSM-Base'
}


class MultiVariantMultiModelResultsAnalyser:
    """Compare multiple dataset variants (GSM8K, main, p1, p2) across many models.

    The class expects a directory where each subdirectory corresponds to a
    dataset variant and contains per-model parquet results. The analyser can
    compute accuracy deltas, run statistical tests and produce plots that
    visualise transitions between baseline and variant results.
    """

    VARIANT_NAME_PATTERN = re.compile(r"(?P<variant>\w+)_test")
    NUMBER_PATTERN = re.compile(r'\d+(?:\.\d+)?')
    BASELINE_VARIANT = 'GSM8K'

    # fixed effect formulations for GLMM
    GLMM1_FIXED_EFFECTS = 'is_variant'
    GLMM2_FIXED_EFFECTS = 'is_variant + gamma_c'

    def __init__(self, dir_path: str | Path, summary_data: pd.DataFrame | None = None, variants: dict | None = None,
                 n_boot: int | None = None):
        self._dir_path = Path(dir_path).resolve()

        if variants is None:
            self._summary_data, self._variants, self._bootstrap_glmm1, self._bootstrap_glmm2 = self._load_data(self._dir_path, n_boot=n_boot)
        else:
            self._summary_data = summary_data
            self._variants = variants
            self._bootstrap_glmm1 = None
            self._bootstrap_glmm2 = None

    @property
    def bootstrap_glmm1(self):
        if self._bootstrap_glmm1:
            df = self._bootstrap_glmm1.summary_df
            df.index.names = ['model', 'metric']

            acc = self.get_accuracy_summary(variant='main', metric='correct').groupby('model').mean()
            acc['metric'] = 'is_variant'
            acc = acc.reset_index().set_index(['model', 'metric'])
            df = pd.concat([df, acc], axis=1, join='outer')

            return df

    @property
    def bootstrap_glmm2(self):
        if self._bootstrap_glmm2:
            df = self._bootstrap_glmm2.summary_df
            df.index.names = ['model', 'metric']
            return df

    @property
    def summary_data(self):
        """Return the concatenated summary DataFrame for all variants."""
        return self._summary_data

    @property
    def variants(self):
        """Return a mapping of variant name -> MultiModelResultsAnalyser."""
        return self._variants

    @property
    def models(self) -> list[str]:
        """List of model identifiers present in the summary data."""
        return self._summary_data.index.tolist()

    @classmethod
    def match_variant_name(cls, name):
        match = cls.VARIANT_NAME_PATTERN.match(name)
        if not match:
            return name
        return match.group('variant')

    @classmethod
    def _load_data(cls, dir_path: Path, n_boot: int | None = None
                   ) -> tuple[
        pd.DataFrame,
        dict[str, MultiModelResultsAnalyser],
        BootstrapResult | None,
        BootstrapResult | None
    ]:

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

        if n_boot is not None:
            boot_path = dir_path.parent / 'bootstrap'
            bootstrap_glmm1 = BootstrapResult(boot_path, n_boot=n_boot, glmm_id='1')
            bootstrap_glmm2 = BootstrapResult(boot_path, n_boot=n_boot, glmm_id='2')
        else:
            bootstrap_glmm1 = None
            bootstrap_glmm2 = None

        return df_summary, variants, bootstrap_glmm1, bootstrap_glmm2

    def _check_variant(self, variant: str):
        if variant not in self._variants:
            raise ValueError(f"No data for variant '{variant}'")

        if variant == self.BASELINE_VARIANT:
            raise ValueError(f"{self.BASELINE_VARIANT} is the baseline variant "
                             f"- choose a different variant to compare it to")

    def get_accuracy_summary(self, variant: str, metric: str | None = None):
        """Return a DataFrame with baseline and variant accuracies and their difference."""
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
        """Return a merged DataFrame comparing variant rows to the baseline.

        The returned DataFrame contains per-example rows with baseline columns
        left-joined on model and template id.
        """
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
        """Plot heatmaps showing transitions between baseline and variant results.

        Produces a pair of heatmaps for numerical correctness and result class
        transitions. Returns a matplotlib Figure containing the two plots.
        """
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

    def prep_glmm1(self, variant: str, metric: str):
        if GLMMRunner is None:
            raise RuntimeError("R not available")

        glmm_runner = GLMMRunner(self.GLMM1_FIXED_EFFECTS)
        data_df = self.get_glmm2_data(variant=variant, metric=metric)
        return glmm_runner, data_df

    def prep_glmm2(self, variant: str, metric: str):
        if GLMMRunner is None:
            raise RuntimeError("R not available")

        glmm_runner = GLMMRunner(self.GLMM2_FIXED_EFFECTS)
        data_df = self.get_glmm2_data(variant=variant, metric=metric)
        return glmm_runner, data_df

    @do_for_metrics
    def run_glmm1(self, variant: str, metric: str, models: list[str] | None = None):
        """Analyse the effect of a dataset variant on accuracy using GLMM.

        Returns a DataFrame with GLMM coefficient estimates and p-values per
        model. If R is not available an error is raised.
        """
        models = self._validate_models(models, variant)

        glmm_runner, data_df = self.prep_glmm1(variant=variant, metric=metric)

        glmm_results_df, diagnostics_df = glmm_runner.run(df=data_df, models=models, simplify=True)

        # add plain accuracy drops
        glmm_results_df = glmm_results_df.join(self.get_mean_accuracy_summary(metric=metric))

        return glmm_results_df, diagnostics_df

    def get_glmm1_data(self, variant: str, metric: str):

        def prep_df(variant_name: str, value: bool):
            data = self.variants[variant_name].full_data
            data = data[['model', 'id', metric]][:]
            data[self.GLMM1_FIXED_EFFECTS] = [value] * len(data)
            data['is_correct'] = data[metric].astype(int)
            data = data.drop(metric, axis=1)
            return data

        baseline_data = prep_df(self.BASELINE_VARIANT, 0)
        variant_data = prep_df(variant, 1)

        df = pd.concat([baseline_data, variant_data]).reset_index(drop=True)
        return df

    def get_glmm2_data(self, variant: str, metric: str):
        number_pattern = re.compile(r'\d*\.?\d+')

        def extract_gamma(text):
            matches = number_pattern.findall(text)
            if not matches:
                return np.nan  # Handle the rare case where a prompt has no numbers
            numbers = (float(match) for match in matches)
            numbers = (number for number in numbers if not number % 1)  # take integers only
            logs = ((np.log10(number) if number > 0 else 0) for number in numbers)
            return sum(logs)

        def _prep(res, variant_label: bool):
            data = res.full_data[['model', 'id', metric, 'question']][:].reset_index(drop=True)
            data[self.GLMM1_FIXED_EFFECTS] = [int(variant_label)] * len(data)
            data['is_correct'] = data[metric]
            data['id'] = data['id']
            data['gamma'] = data.question.apply(extract_gamma)

            data = data.drop(metric, axis=1).drop('question', axis=1)
            return data

        baseline_df = _prep(self.variants[self.BASELINE_VARIANT], False)
        variant_df = _prep(self.variants[variant], True)

        df = pd.concat([baseline_df, variant_df]).reset_index(drop=True)
        df['gamma_c'] = df['gamma'] - df['gamma'].mean()
        df = df.drop('gamma', axis=1)
        return df

    @do_for_metrics
    def run_glmm2(self, variant: str, metric: str, models: list[str] | None = None):
        """Analyse the effect of numeric quantities in questions on correctness.

        Fits a GLMM that includes a covariate derived from the log10 of numbers
        appearing in the question text (sum of logs of integer tokens).
        """
        if models is not None:
            models = self._validate_models(models, variant)

        glmm_runner, data_df = self.prep_glmm2(variant=variant, metric=metric)

        glmm_results_df, diagnostics_df = glmm_runner.run(df=data_df, models=models)
        return glmm_results_df, diagnostics_df

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
            variant_label = VARIANT_LABELS.get(variant_name, variant_name)
            variant_df = self.variants[variant_name].full_data
            model_df = variant_df.loc[variant_df['model'] == (model or self.models[0]), ['question']].copy()
            extracted_numbers = model_df['question'].str.findall(self.NUMBER_PATTERN).explode().dropna().astype(float)
            extracted_numbers = extracted_numbers[~(extracted_numbers % 1).astype(bool)]  # limit to integers
            raw_counts_dict[variant_label] = Counter(extracted_numbers)

            # put into bins, all fractions in a single separate bin
            binned = pd.cut(extracted_numbers, bins=bin_edges, labels=bin_labels, right=False, include_lowest=True)
            number_counts = binned.value_counts().reindex(bin_labels, fill_value=0)
            binned_counts_dict[variant_label] = number_counts

        raw_counts_df = pd.DataFrame(raw_counts_dict).fillna(0).astype(int).sort_index()
        binned_counts_df = pd.DataFrame(binned_counts_dict).fillna(0).astype(int)

        return raw_counts_df, binned_counts_df

    def plot_number_counts(self, model: str | None = None, bin_edges: list[int | float] | None = None, **kwargs):
        raw_counts_df, binned_counts_df = self.get_number_counts(model=model, bin_edges=bin_edges)
        return plot_number_counts(raw_counts_df, binned_counts_df, **kwargs)

    def get_clean_data_object(self):
        return MultiVariantMultiModelResultsAnalyser(
            dir_path=self._dir_path,
            summary_data=None,
            variants={k: v.get_clean_data_object() for k, v in self._variants.items()}
        )

