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
            return combined[['significant', 'success', 'failure']].astype(int).groupby('param').sum()

    def plot_core_stats(self, variant: str, title: str | None = None, **kwargs):
        titles = {'babbling': 'Babbling factor', 'correct': 'Accuracy (standard)', 'correct_strict': 'Accuracy (discounted)'}
        colors =['limegreen', 'indianred', 'lightsteelblue']

        cs = self.compare_core_stats(variant, **kwargs, detailed_output=False)
        n_models = len(self._experiment_mres.variants[variant].models)
        cs['not significant'] = n_models - cs.significant
        cs = cs.drop('significant', axis=1)
        cs.rename(columns={'success': 'Improvement', 'failure': 'Deterioration', 'not significant': 'Change not significant'})

        n_plots = len(cs)
        fig, axes = plt.subplots(1, n_plots + 1, figsize=(12, 4), gridspec_kw={'width_ratios': [2, 2, 2, 1]})
        for param, ax in zip(cs.index, axes):
            wedges, _, _ = ax.pie(cs.loc[param], colors=colors, autopct=lambda p: str(round(p*n_models/100)))
            ax.set_title(titles.get(param, param))

        axes[-1].axis('off')
        axes[-1].legend(wedges, cs.columns, loc='center')

        fig.suptitle(title or self._experiment_label + f" ('{variant}' variant)")

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

    def analyze_accuracy_drops(self, variant: str, alpha=0.05):
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

        return combined
