from scipy import stats
import pandas as pd
import logging

from gsm_benchmarker.results_analyser import MultiVariantMultiModelResultsAnalyser
from gsm_benchmarker.results_analyser.plotting_utils import plot_stats
from gsm_benchmarker.results_analyser.common import GLMMRunner

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

    def analyse_accuracy_change_significance(self, variant: str = 'main', models: list[str] | None = None
                                             , metric: str | None = None):
        """
        Run two-tailed GLMM test (per model) to check whether accuracy change between experiment and baseline
        on a given variant is significant.
        """

        glmm_runner = GLMMRunner(
            label='is_experiment',
            question_difficulties=self._baseline_mres.get_question_difficulty_per_model()
        )

        if models is None:
            models = list(set(self._experiment_mres.models) + set(self._baseline_mres.models))

        models_validated = []
        for model in models:
            if model not in self._experiment_mres.models or model not in self._baseline_mres.models:
                logger.warning(f"No data for model {model}")
            else:
                models_validated.append(model)

        glmm_results_df = glmm_runner.run(
            ras={
                0: self._baseline_mres.variants[variant],
                1: self._experiment_mres.variants[variant]
            },
            models=models_validated,
            metric=metric
        )

        # add plain accuracy change
        acc_change = self.get_raw_acc_change(variant=variant, metric=metric)
        gb = ['model', 'metric'] if metric is None else ['model']
        glmm_results_df['mean_diff'] = acc_change.groupby(gb).mean()
        glmm_results_df['median_diff'] = acc_change.groupby(gb).median()

        return glmm_results_df

    def get_raw_acc_change(self, variant: str = 'main', metric: str | None = None) -> pd.DataFrame:
        baseline_accuracies = self._baseline_mres.variants[variant].get_accuracies_per_model_and_template_id(metric=metric)
        experiment_accuracies = self._experiment_mres.variants[variant].get_accuracies_per_model_and_template_id(metric=metric)
        acc_change = experiment_accuracies - baseline_accuracies
        acc_change.rename('acc_change', inplace=True)
        return acc_change
