from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from tqdm.auto import tqdm

from gsm_benchmarker.results_analyser import MultiVariantMultiModelResultsAnalyser


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

        return self.plot_stats(cs, n_models=n_models, titles=titles,
                               title=title or f"{self._experiment_label} - per-model performance improvement on '{variant}' variant)")

    def plot_stats(self, cs: pd.DataFrame, n_models: int = 20, titles: dict | None = None, title: str | None = None):
        colors =['limegreen', 'indianred', 'lightsteelblue']

        cs['not significant'] = n_models - cs.significant
        cs = cs.drop('significant', axis=1)
        cs = cs.rename(columns={'success': 'Improvement', 'failure': 'Deterioration', 'not significant': 'Change not significant'})

        n_plots = len(cs)
        fig, axes = plt.subplots(1, n_plots + 1, figsize=(12, 4))
        for param, ax in zip(cs.index, axes):
            wedges, _, _ = ax.pie(cs.loc[param], colors=colors, autopct=lambda p: str(round(p*n_models/100)))
            ax.set_title(titles.get(param, param))

        axes[-1].axis('off')
        axes[-1].legend(wedges, cs.columns, loc='center', title="No. if models which showed:")

        if title:
            fig.suptitle(title)

        return fig, cs

    def _analyse_single_model_accuracy_drop(self, base_data: pd.Series, exp_data: pd.Series, alpha=0.05):
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

        drops_base = self._baseline_mres.get_accuracy_drop_df(variant)
        drops_experiment = self._experiment_mres.get_accuracy_drop_df(variant)

        # Get list of unique models from the index level 0
        models = drops_base.index.get_level_values(0).unique()

        res = {}

        for model in tqdm(models):
            model_res = {}
            for col_name in drops_experiment.columns:
                try:
                    base_data = drops_base.loc[model][col_name].sort_index()
                    exp_data = drops_experiment.loc[model][col_name].sort_index()
                except KeyError:
                    logger.warning(f"Skipping {model}: data missing in one of the dataframes")
                    continue

                # Check alignment
                if not base_data.index.equals(exp_data.index):
                    logger.warning(f"Template IDs do not match for {model}.; skipping")
                    continue

                model_res[col_name] = self._analyse_single_model_accuracy_drop(base_data, exp_data, alpha=alpha)

            res[model] = pd.DataFrame(model_res).T

        combined = pd.concat(res.values(), keys=res.keys(), names=['model', 'metric'])

        if detailed_output:
            return combined
        else:
            return self._summarise_output(combined, 'metric')

    def plot_gap_closure_bars(self, variant, alpha: float = 0.05, projected_alpha: float | None = None, **kwargs):
        cs = self.analyze_gap_closure(variant, **kwargs, detailed_output=True)
        cs.rename(
            index={'accuracy_drop': 'Standard accuracy', 'strict_accuracy_drop': 'Discounted accuracy'},
            level='metric',
            inplace=True
        )

        fig = self._plot_bars_and_p_bars(
            cs, 'mean_gap_closure', 'p_value', alpha=alpha, projected_alpha=projected_alpha,
            title=f"{self._experiment_label}: accuracy (gap closure) on variant '{variant}' vs 'GSM8K'"
        )

        return fig

    @staticmethod
    def _plot_bars_and_p_bars(df: pd.DataFrame, value_col: str, p_value_col: str,
                              alpha: float = 0.05, projected_alpha: float | None = None, title: str | None = None,
                              colours: list[str] | None = None):

        if colours is None:
            colours = ['lightblue', 'navy']

        def prep_data(col):
            d = df[col].unstack(level='metric')
            d = d[['Standard accuracy', 'Discounted accuracy']]
            return d

        df_gap_closure = prep_data(value_col)
        df_p_values = prep_data(p_value_col)

        fig, axes = plt.subplots(2, 1, sharex='all', figsize=(12, 8))

        df_gap_closure.plot(ax=axes[0], kind='bar', color=colours)
        axes[0].set_ylabel(value_col.replace('_', ' ').capitalize())
        axes[0].axhline(0, color='k', lw=0.5)

        df_p_values.plot(ax=axes[1], kind='bar', color=colours)
        axes[1].set_xticklabels(['_'.join(s.split('_')[1:]) for s in df_p_values.index], rotation=45, ha='right')
        axes[1].axhline(alpha, ls='--', color='k', lw=0.5, label='alpha = 0.05')

        if projected_alpha is not None:
            axes[1].axhline(projected_alpha, ls=':', color='maroon', lw=0.5, label='equivalent alpha for full set')

        axes[1].set_xlabel('Model')
        axes[1].set_ylabel('P value')

        for ax in axes:
            ax.legend(loc='upper right', frameon=True)
            for container in ax.containers:
                ax.bar_label(container, fmt="%.2f", fontsize=7)

        if title:
            fig.suptitle(title)

        fig.tight_layout()

        return fig

    def plot_gap_closure_summary(self, variant: str, title: str | None = None, **kwargs):
        titles = {'accuracy_drop': 'Standard accuracy', 'strict_accuracy_drop': 'Discounted accuracy'}

        cs = self.analyze_gap_closure(variant, **kwargs, detailed_output=False)
        n_models = len(self._experiment_mres.variants[variant].models)

        fig, cs = self.plot_stats(
            cs, n_models=n_models, titles=titles,
            title=title or f"{self._experiment_label}: accuracy drop (gap closure) on variant '{variant}' vs 'GSM8K'")

        fig.subplots_adjust(top=0.8)
        return fig, cs

    def analyse_gap_significance(self, variant: str ='main'):
        def prep_frame(df, label):
            df = df.set_index('model')
            df.rename(columns={c: f"{label}_{c}" for c in df.columns}, inplace=True)
            return df

        res = {}
        for metric, metric_label in (('correct', 'standard'), ('correct_strict', 'discounted')):
            baseline_gaps = self._baseline_mres.run_gap_analysis(metric=metric, variant=variant)
            experiment_gaps = self._experiment_mres.run_gap_analysis(metric=metric, variant=variant)

            res[metric_label] = prep_frame(baseline_gaps, 'baseline').join(prep_frame(experiment_gaps, 'experiment'))

        df_results = pd.concat(res.values(), keys=res.keys(), names=('metric', 'model'))
        df_results = df_results.swaplevel().sort_index()
        return df_results

    def plot_gap_significance_bars(self, variant: str = 'main', alpha: float = 0.05, projected_alpha: float | None = None, **kwargs):
        df = self.analyse_gap_significance(variant)
        df.rename(
            index={'standard': 'Standard accuracy', 'discounted': 'Discounted accuracy'},
            level='metric',
            inplace=True
        )

        fig1 = self._plot_bars_and_p_bars(
            df, 'baseline_gap', 'baseline_p_value', alpha=alpha, projected_alpha=projected_alpha,
            title=f"Baseline (GSM-Symbolic): accuracy drop on variant '{variant}' vs 'GSM8K'"
        )

        fig2 = self._plot_bars_and_p_bars(
            df, 'experiment_gap', 'experiment_p_value', alpha=alpha, projected_alpha=projected_alpha,
            title=f"{self._experiment_label}: accuracy drop on variant '{variant}' vs 'GSM8K'"
        )

        ymax = [max(fig1.axes[i].get_ylim()[1], fig2.axes[i].get_ylim()[1]) for i in range(2)]

        for fig in (fig1, fig2):
            fig.axes[0].set_ylabel('Accuracy drop')
            for i, ym in enumerate(ymax):
                fig.axes[i].set_ylim(fig.axes[i].get_ylim()[0], ym)

        return fig1, fig2

    def run_variant_accuracy_analysis(self, metric: str = 'correct', variant: str = 'main'):
        """
        Run one-tailed Wilcoxon signed-rank test (per model) to check whether accuracy change betw experiment
        and baseline on a given variant is significant.
        """

        df_baseline = self._baseline_mres.variants[variant].full_data
        df_experiment = self._experiment_mres.variants[variant].full_data

        results = []

        for model in df_baseline.model.unique():
            if model not in df_experiment.model.unique():
                continue

            # filter by model, aggregate by template id
            scores_baseline = df_baseline[df_baseline.model == model].groupby('id')[metric].mean()
            scores_experiment = df_experiment[df_experiment.model == model].groupby('id')[metric].mean()

            median_diff = np.median(scores_experiment - scores_baseline)

            if median_diff:
                stat, p_value = stats.wilcoxon(scores_baseline, scores_experiment, alternative='two-sided')
            else:
                stat = 0
                p_value = 1

            results.append({'model': model, 'median_diff': median_diff, 'p_value': p_value, 'stat': stat})

        df = pd.DataFrame(results)
        df.set_index('model', inplace=True)
        return df

    def analyse_accuracy_change_significance(self, variant: str ='main'):
        res = {}
        for metric, metric_label in (('correct', 'standard'), ('correct_strict', 'discounted')):
            res[metric_label] = self.run_variant_accuracy_analysis(metric=metric, variant=variant)

        df_results = pd.concat(res.values(), keys=res.keys(), names=('metric', 'model'))
        df_results = df_results.swaplevel().sort_index()
        return df_results


    def plot_accuracy_change_significance_bars(self, variant: str = 'main', alpha: float = 0.05, projected_alpha: float | None = None, **kwargs):

        df = self.analyse_accuracy_change_significance(variant)
        df.rename(
            index={'standard': 'Standard accuracy', 'discounted': 'Discounted accuracy'},
            level='metric',
            inplace=True
        )

        fig = self._plot_bars_and_p_bars(
            df, 'median_diff', 'p_value', alpha=alpha, projected_alpha=projected_alpha,
            title=f"Change in accuracy from baseline (GSM8K) to {self._experiment_label}, variant '{variant}'",
            colours=['lightgreen', 'seagreen']
        )

        fig.axes[0].set_ylabel('Accuracy change')

        return fig
