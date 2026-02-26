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
from statsmodels.stats.multitest import multipletests

from rpy2.rinterface_lib.embedded import RRuntimeError
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

# Set pandas converter as the global default for rpy2
ro.conversion.set_conversion(pandas2ri.converter + ro.default_converter)

from pymer4.models import glmer  # needs to go after the converter setting

from gsm_benchmarker.results_analyser.multi_model import MultiModelResultsAnalyser
from gsm_benchmarker.results_analyser.plotting_utils import plot_question_success_rate_matrix


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

        for model in self.models:
            difficulties[model] = self.get_question_difficulty(model=model)

        difficulties['OVERALL'] = self.get_question_difficulty()

        difficulties_df = pd.DataFrame(difficulties).T
        return difficulties_df

    def plot_question_difficulty_per_model(self):
        difficulties = self.get_question_difficulty_per_model()
        return plot_question_success_rate_matrix(difficulties)

    def plot_question_difficulty_histogram(self, model: str | None = None):
        difficulties = self.get_question_difficulty(model=model)

        fig, ax = plt.subplots()
        ax.hist(difficulties, 21)
        ax.set_xlabel("Overall question difficulty")
        ax.set_ylabel("Question count")
        return fig

    @classmethod
    def _fit_glmm(cls, df):
        glmm_model = glmer(
            'is_correct ~ is_variant + difficulty + (1 | id)',
            data=df,
            family='binomial'
        )

        try:
            glmm_model.fit(verbose=False)  # fitting works, only getting stats fails
        except RRuntimeError as err:
            if glmm_model.r_model is None:
                raise RuntimeError(f"GLMM fitting failed: {err}")

        # Assign the model to an R variable first
        ro.globalenv['r_model'] = glmm_model.r_model

        # Then extract coefficients as a DataFrame
        with localconverter(ro.default_converter + pandas2ri.converter):
            coefs_df = ro.r('as.data.frame(coef(summary(r_model)))')

        res = dict(
            estimate=coefs_df.loc['is_variant', 'Estimate'],
            p_value=coefs_df.loc['is_variant', 'Pr(>|z|)'],
            std_err=coefs_df.loc['is_variant', 'Std. Error'],
        )

        return res

    def _prep_df_for_glmm(self, variant: str, metric: str):
        def prep_df(df_variant):
            res = self.variants[df_variant].full_data
            res = res[['model', 'id', metric]][:]
            res['is_variant'] = [int(df_variant != 'GSM8K')] * len(res)
            res['is_correct'] = res[metric].astype(int)
            res = res.drop(metric, axis=1)
            return res

        df = pd.concat([prep_df(self.BASELINE_VARIANT), prep_df(variant)]).reset_index(drop=True)
        return df

    def analyse_variant_effect_glmm(self, variant: str):
        res = {}
        for metric, metric_label in (('correct', 'standard'), ('correct_strict', 'discounted')):
            res[metric_label] = self._analyse_metric_variant_effect_glmm(variant=variant, metric=metric)

        df_results = pd.concat(res.values(), keys=res.keys(), names=('metric', 'model'))
        df_results = df_results.swaplevel().sort_index()
        return df_results

    def _analyse_metric_variant_effect_glmm(self, variant: str, metric: str):
        df = self._prep_df_for_glmm(variant, metric=metric)
        glmm_results = []

        for model_name, group_df in df.groupby('model'):
            difficulty = self.get_question_difficulty(model=model_name)
            group_df = group_df.merge(difficulty.reset_index(), on='id', how='left')

            try:
                res = self._fit_glmm(group_df)
            except RuntimeError as err:
                logger.warning(f"{model_name}: {err}")
                continue

            glmm_results.append({
                'model': model_name,
                **res,
            })

        glmm_results_df = pd.DataFrame(glmm_results)
        self._enrich_glmm_summary(glmm_results_df, metric=('strict_' if 'strict' in metric else '') + 'accuracy_drop')
        glmm_results_df = glmm_results_df.set_index('model')
        return glmm_results_df

    def _enrich_glmm_summary(self, glmm_results_df: pd.DataFrame, metric: str):
        model_accuracy_drops = self.get_accuracy_drop_df('main').groupby('model').mean().reset_index()
        glmm_results_df['accuracy_drop'] = model_accuracy_drops[metric]

        estimate = glmm_results_df.estimate
        glmm_results_df['drop'] = estimate < 0
        glmm_results_df['odds_ratio'] = np.exp(estimate)

        # 95% Confidence Intervals in log-odds and odds ratios
        std_err_clipped = np.minimum(glmm_results_df['std_err'], estimate.abs().max())  # clip - for plotting
        ci_lower_log = estimate - 1.96 * std_err_clipped
        ci_upper_log = estimate + 1.96 * std_err_clipped
        glmm_results_df['ci_lower_or'] = np.exp(ci_lower_log)
        glmm_results_df['ci_upper_or'] = np.exp(ci_upper_log)

        # apply Benjamini-Hochberg procedure - controls the false discovery rate (multiple comparisons correction)
        rejected, p_corrected, _, _ = multipletests(glmm_results_df['p_value'], method='fdr_bh')
        glmm_results_df['p_value_corrected'] = p_corrected

        return glmm_results_df

