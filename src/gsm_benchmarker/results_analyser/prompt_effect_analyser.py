from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from gsm_benchmarker.results_analyser import MultiVariantMultiModelResultsAnalyser


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
                print(f"Model '{model}' not found in new results")
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
