from scipy import stats
import numpy as np
import pandas as pd
import logging
from tqdm.auto import tqdm

from gsm_benchmarker.results_analyser import MultiVariantMultiModelResultsAnalyser
from gsm_benchmarker.results_analyser.plotting_utils import plot_stats, plot_bars_and_p_bars
from gsm_benchmarker.results_analyser.common import do_for_metrics, GLMMRunner

logger = logging.getLogger(__name__)


class PromptEffectAnalyser:
    def __init__(
            self,
            baseline_mres: MultiVariantMultiModelResultsAnalyser,
            experiment_mres: MultiVariantMultiModelResultsAnalyser,
            experiment_label: str | None = None
    ):
        self._baseline_mres = baseline_mres
        self._experiment_mres = experiment_mres
        self._experiment_label = experiment_label


    def compare_core_stats(self, variant: str, alpha=0.05, detailed_output: bool = False):

        orig = self._baseline_mres.variants[variant].get_core_stats()
        new = self._experiment_mres.variants[variant].get_core_stats()

        orig_models = orig.index.unique(level=0)
        new_models = new.index.unique(level=0)

        res = {}
        want_increase = {'correct': True, 'correct_strict': True, 'babbling': False}

        for model in orig_models:

            if model not in new_models:
                logger.warning(f"Model '{model}' not found in new results")
                continue

            r = {}
            for column in orig.columns:
                u = orig.loc[model][column]
                v = new.loc[model][column]
                t_stat, p_value = stats.ttest_rel(v, u)
                rc = {}
                rc['mean_diff'] = v.mean() - u.mean()
                rc["p_value"] = p_value
                rc['t_stat'] = t_stat

                significant = p_value < alpha
                good_change = t_stat > 0 if want_increase[column] else t_stat < 0
                rc['significant'] = significant
                rc['success'] = significant and good_change
                rc['failure'] = significant and not good_change

                r[column] = rc

            res[model] = pd.DataFrame(r).T

        combined = pd.concat(res.values(), keys=res.keys(), names=['model', 'param'])

        if detailed_output:
            return combined
        else:
            return self._summarise_output(combined, 'param')

    @staticmethod
    def _summarise_output(combined_df, column):
        return combined_df[['significant', 'success', 'failure']].astype(int).groupby(column).sum()

    def plot_core_stats(self, variant: str, title: str | None = None, **kwargs):
        titles = {'babbling': 'Babbling factor', 'correct': 'Accuracy (standard)', 'correct_strict': 'Accuracy (discounted)'}

        cs = self.compare_core_stats(variant, **kwargs, detailed_output=False)
        n_models = len(self._experiment_mres.variants[variant].models)

        return plot_stats(cs, n_models=n_models, titles=titles,
                          title=title or f"{self._experiment_label} - per-model performance improvement on '{variant}' variant)")

    @staticmethod
    def _analyse_single_model_accuracy_drop(base_data: pd.Series, exp_data: pd.Series, alpha=0.05):
        # calculate the 'gap closure' (difference of differences)
        # we want to test if baseline drop > experiment drop, so: gap closure = baseline - experiment
        diffs = base_data - exp_data

        # edge case - no difference
        if np.all(diffs == 0):
            return {
                'mean_gap_closure': 0.0,
                'test_type': 'None (Identical)',
                'p_value': 1.0,
                'significant': False,
                'success': False,
                'failure': False
            }

        # check normality (Shapiro-Wilk)
        shapiro_stat, shapiro_p = stats.shapiro(diffs)
        is_normal = shapiro_p > 0.05

        # run the statistical test
        # use 'greater' because we expect baseline drop > treatment drop
        if is_normal:
            test_name = 'Paired T-Test'
            stat, p_val = stats.ttest_rel(base_data, exp_data, alternative='greater')
        else:
            test_name = 'Wilcoxon'
            # 'wilcoxon' excludes zero-differences by default (correction=True)
            stat, p_val = stats.wilcoxon(base_data, exp_data, alternative='greater')

        significant = p_val < alpha
        good_change = stat > 0

        return {
            'mean_gap_closure': diffs.mean(),
            'test_type': test_name,
            'p_value': p_val,
            'significant': significant,
            'success': significant and good_change,
            'failure': significant and not good_change
        }

    def analyze_gap_closure(self, variant: str, alpha=0.05, detailed_output=False):
        """Analyses whether the accuracy drop is significantly lower in the treatment group."""

        drops_base = self._baseline_mres.get_accuracy_drops(variant)
        drops_experiment = self._experiment_mres.get_accuracy_drops(variant)

        # Get list of unique models from the index level 0
        models = drops_base.index.get_level_values(0).unique()

        res = {}

        for model in tqdm(models):
            model_res = {}
            for metric_name in drops_experiment.index.get_level_values('metric').unique():
                try:
                    base_data = drops_base.loc[model].xs(metric_name, level='metric').sort_index()
                    exp_data = drops_experiment.loc[model].xs(metric_name, level='metric').sort_index()
                except KeyError:
                    logger.warning(f"Skipping {model}: data missing in one of the dataframes")
                    continue

                # Check alignment
                if not base_data.index.equals(exp_data.index):
                    logger.warning(f"Template IDs do not match for {model}.; skipping")
                    continue

                model_res[metric_name] = self._analyse_single_model_accuracy_drop(base_data, exp_data, alpha=alpha)

            res[model] = pd.DataFrame(model_res).T

        combined = pd.concat(res.values(), keys=res.keys(), names=['model', 'metric'])

        if detailed_output:
            return combined
        else:
            return self._summarise_output(combined, 'metric')

    def plot_gap_closure_bars(self, variant, alpha: float = 0.05, projected_alpha: float | None = None, **kwargs):
        cs = self.analyze_gap_closure(variant, **kwargs, detailed_output=True)

        fig = plot_bars_and_p_bars(
            cs, 'mean_gap_closure', 'p_value', alpha=alpha, projected_alpha=projected_alpha,
            title=f"{self._experiment_label}: accuracy (gap closure) on variant '{variant}' vs 'GSM8K'"
        )

        return fig

    def plot_gap_closure_summary(self, variant: str, title: str | None = None, **kwargs):
        titles = {'accuracy_drop': 'Standard accuracy', 'strict_accuracy_drop': 'Discounted accuracy'}

        cs = self.analyze_gap_closure(variant, **kwargs, detailed_output=False)
        n_models = len(self._experiment_mres.variants[variant].models)

        fig, cs = plot_stats(
            cs, n_models=n_models, titles=titles,
            title=title or f"{self._experiment_label}: accuracy drop (gap closure) on variant '{variant}' vs 'GSM8K'")

        fig.subplots_adjust(top=0.8)
        return fig, cs

    @do_for_metrics
    def analyse_gap_significance(self, variant: str, metric: str):
        def prep_frame(df, label):
            df = df.set_index('model')
            df.rename(columns={c: f"{label}_{c}" for c in df.columns}, inplace=True)
            return df

        baseline_gaps = self._baseline_mres.run_gap_analysis(metric=metric, variant=variant)
        experiment_gaps = self._experiment_mres.run_gap_analysis(metric=metric, variant=variant)

        res = prep_frame(baseline_gaps, 'baseline').join(prep_frame(experiment_gaps, 'experiment'))

        return res

    def plot_gap_significance_bars(self, variant: str = 'main', alpha: float = 0.05,
                                   projected_alpha: float | None = None, only_significant_models: bool = False):
        df = self.analyse_gap_significance(variant)

        fig1 = plot_bars_and_p_bars(
            df, 'baseline_gap', 'baseline_p_value', alpha=alpha, projected_alpha=projected_alpha,
            title=f"Baseline (GSM-Symbolic): accuracy drop on variant '{variant}' vs 'GSM8K'"
        )

        if only_significant_models:
            significant_df = df[df.baseline_p_value < (projected_alpha if projected_alpha is not None else alpha)]
            significant_models = list(significant_df.index.get_level_values('model').unique())
        else:
            significant_models = None

        fig2 = plot_bars_and_p_bars(
            df, 'experiment_gap', 'experiment_p_value', alpha=alpha, projected_alpha=projected_alpha,
            title=f"{self._experiment_label}: accuracy drop on variant '{variant}' vs 'GSM8K'",
            models=significant_models
        )

        ymax = [max(fig1.axes[i].get_ylim()[1], fig2.axes[i].get_ylim()[1]) for i in range(2)]

        for fig in (fig1, fig2):
            fig.axes[0].set_ylabel('Accuracy drop')
            for i, ym in enumerate(ymax):
                fig.axes[i].set_ylim(fig.axes[i].get_ylim()[0], ym)

        return fig1, fig2, df

    def _prep_df_for_glmm(self, variant: str, metric: str):
        def prep_df(mres: MultiVariantMultiModelResultsAnalyser, is_experiment: bool):
            res = mres.variants[variant].full_data
            res = res[['model', 'id', metric]][:]
            res['is_variant'] = [is_experiment] * len(res)
            res['is_correct'] = res[metric].astype(int)
            res = res.drop(metric, axis=1)
            return res

        df = pd.concat(
            [
                prep_df(self._baseline_mres, False),
                prep_df(self._experiment_mres, True)
            ]
        ).reset_index(drop=True)

        return df

    def analyse_accuracy_change_significance(self, variant: str = 'main', models: list[str] | None = None):
        """
        Run two-tailed GLMM test (per model) to check whether accuracy change between experiment and baseline
        on a given variant is significant.
        """

        glmm_runner = GLMMRunner(
            label='is_experiment',
            question_difficulties=self._baseline_mres.get_question_difficulty_per_model()
        )

        glmm_results_df = glmm_runner.run(
            ras={
                0: self._baseline_mres.variants[variant],
                1: self._experiment_mres.variants[variant]
            },
            models=models
        )

        # add plain accuracy change
        baseline_accuracies = self._baseline_mres.variants[variant].get_accuracies_per_model_and_template_id()
        experiment_accuracies = self._experiment_mres.variants[variant].get_accuracies_per_model_and_template_id()
        acc_change = experiment_accuracies - baseline_accuracies
        glmm_results_df['mean_diff'] = acc_change.groupby(['model', 'metric']).mean()
        glmm_results_df['median_diff'] = acc_change.groupby(['model', 'metric']).median()

        return glmm_results_df

    def plot_accuracy_change_significance_bars(self, variant: str = 'main', alpha: float = 0.05, projected_alpha: float | None = None, **kwargs):

        df = self.analyse_accuracy_change_significance(variant=variant)

        fig = plot_bars_and_p_bars(
            df, 'median_diff', 'p_value', alpha=alpha, projected_alpha=projected_alpha,
            title=f"Change in accuracy from baseline (GSM8K) to {self._experiment_label}, variant '{variant}'",
            colours=['lightgreen', 'seagreen'], **kwargs
        )

        fig.axes[0].set_ylabel('Accuracy change')

        return fig, df
