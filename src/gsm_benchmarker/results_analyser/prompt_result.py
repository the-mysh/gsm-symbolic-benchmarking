"""Utilities that group prompt-level results and present summaries/plots.

This module provides the PromptResult dataclass representing a single
prompt-format evaluation and a small helper to aggregate multiple prompts
for side-by-side comparison and LaTeX export.
"""

from pathlib import Path
import pandas as pd
from dataclasses import dataclass
import numpy as np
from functools import cached_property
from statsmodels.stats.multitest import multipletests

from gsm_benchmarker.results_analyser import MultiVariantMultiModelResultsAnalyser
from gsm_benchmarker.results_analyser.prompt_effect_analyser import PromptEffectAnalyser
from gsm_benchmarker.results_analyser.plotting_utils import (plot_glmm, plot_acc_change_distribution, Colour,
                                                             plot_prompt_comparison, plot_prompt_acc_evolution)
from gsm_benchmarker.results_analyser.utils import pandas_to_latex, correct_p_values, format_p_value, format_float


@dataclass
class PromptResult:
    """Container for results and plotting helpers for a single prompt format.

    Attributes mirror the constructor parameters: path points to the directory
    with variant subdirectories, colour is a Colour instance used for plotting,
    and mres holds a MultiVariantMultiModelResultsAnalyser for the prompt.
    """
    path: str | Path
    colour: Colour
    full_label: str
    short_label: str = None
    models: list[str] | None = None
    metric: str = None
    save_dest: Path | None = None
    mres: MultiVariantMultiModelResultsAnalyser = None
    baseline: MultiVariantMultiModelResultsAnalyser | None = None
    pea: PromptEffectAnalyser | None = None
    n_boot: int | None = None

    def __post_init__(self):
        if self.mres is None:
            self.mres = MultiVariantMultiModelResultsAnalyser(self.path, n_boot=self.n_boot)

        if self.baseline is not None and self.pea is None:
            assert self.mres is not None
            self.pea = PromptEffectAnalyser(self.baseline, self.mres, self.full_label)

        if self.short_label is None:
            self.short_label = self.full_label.split(' ')[0]

    def _filter_glmm_result_df(self, bs: pd.DataFrame | None) -> pd.DataFrame | None:
        if bs is None:
            return
        if self.models:
            bs = bs.loc[bs.index.isin(self.models, level=0)]
        return bs

    @cached_property
    def glmm1_results(self) -> pd.DataFrame | None:
        return self._filter_glmm_result_df(self.mres.bootstrap_summary_glmm1)

    def glmm1_results_to_latex(self, alpha=0.05, projected_alpha: float | None = None,
                               model_order: list[str] | None = None, position: str = "H"):
        df = self.glmm1_results.copy().xs('is_variant', level='metric')

        df['boot_odds_ratio'] = np.exp(df['boot_median_log'])
        df['boot_ci_upper'] = np.exp(df['boot_ci_upper_log'])
        df['boot_ci_lower'] = np.exp(df['boot_ci_lower_log'])

        if self.models is not None:
            df = df[df.index.isin(self.models, level='model')]

        if model_order is not None:
            df = df.sort_index(
                key=lambda c: c.map({model: index for index, model in enumerate(model_order)})
            )

        df1 = pd.DataFrame({
            'GSM-Base acc': df['GSM8K_acc'].apply(format_float(1)),
            'GSM-Variants acc': df['main_acc'].apply(format_float(1)),
            r'$\Delta$ Acc': df['acc_diff'].apply(format_float(2)),
            'P value':  df.apply(lambda row: f"{format_p_value(3)(row['boot_p_value'])}{' $\\ddagger$' if not row['agreement'] else ''}", axis=1),
            'Corrected P value': correct_p_values(df['boot_p_value']).apply(format_p_value(3))
        }, index=df.index)
        df1.index.name = 'Model'


        df2 = pd.DataFrame({
            'Odds ratio': df.apply(lambda row: f"{row['boot_odds_ratio']:.2f}", axis=1),
            r'95\% CI': df.apply(lambda row: f"[{row['boot_ci_lower']:.2f}, {row['boot_ci_upper']:.2f}]", axis=1),
            r'Rand. eff. variance $\pm$ std. dev.': df.apply(
                lambda row: (
                        f"{row['wald_ranef_variance']:.2f} " + r"$\pm$" + f" {row['wald_ranef_sd']:.2f}"
                ), axis=1),
            'N estimates': df['boot_n_clean'].apply(format_float(0)),
        }, index=df.index)
        df2.index.name = 'Model'

        caption1 = f"Results of {self.full_label}"
        if not df['agreement'].all():
            caption1 += (" The $\ddagger$ symbol marks cases where significance estimated with Wald method "
                         "did not match the bootstrap estimate.")

        df1_latex = pandas_to_latex(df1, label=f"tab:{self.short_label}-results", caption=caption1,
                                    clean_header=False, position=position)

        caption2 = f"Additional statistics for results of {self.full_label}."
        df2_latex = pandas_to_latex(df2, label=f"tab:{self.short_label}-stats", caption=caption2,
                                    clean_header=False, position=position)

        print(df1_latex)
        print(df2_latex)

    def _check_pea(self):
        if self.pea is None:
            raise ValueError(f"Prompt effect analysis not possible for baseline prompt ({self.full_label})")
        return self.pea

    @cached_property
    def glmm2_results(self) -> pd.DataFrame | None:
        return self._filter_glmm_result_df(self.mres.bootstrap_summary_glmm2)

    def plot_glmm1(self, **kwargs):
        """Produce GLMM visualisations for the variant effect of this prompt."""
        figs = plot_glmm(
            self.glmm1_results,
            'acc_diff',
            "Variant performance delta, pp",
            bar_colour=self.colour.value,
            save_prefix=self.save_dest/self.short_label if self.save_dest is not None else None,
            **kwargs
        )

        return figs

    def plot_acc_change_dist(self, **kwargs):
        acc_change_raw = self._check_pea().get_accuracy_change(variant='main', metric=self.metric)

        fig = plot_acc_change_distribution(
            acc_change_raw,
            label="Prompt performance delta, pp",
            models=self.models,
            color=self.colour.value,
            save_prefix=self.save_dest/self.short_label if self.save_dest is not None else None,
            **kwargs
        )

        return fig

    def get_significant_models(self, alpha: float, drop_only: bool = False):
        df = self.glmm1_results
        if df is None:
            return None
        if drop_only:
            df = df[df.acc_diff < 0]
        models = df[df.boot_p_value < alpha].sort_values('boot_median_log', ascending=True).index.get_level_values('model').tolist()
        return models

    def summary(self, alpha: float = 0.05):

        g1 = self.glmm1_results.xs('is_variant', level='metric')

        d = {
            'GSM8K_acc': self.mres.variants['GSM8K'].get_accuracies_per_model(metric=self.metric),
            'main_acc': self.mres.variants['main'].get_accuracies_per_model(metric=self.metric),
            'acc_diff': g1['acc_diff'],
            'variant_effect_log_or': g1['boot_median_log'],
            'variant_effect_or': np.exp(g1['boot_median_log']),
            'variant_effect_p_value': g1['boot_p_value'],
            'variant_effect_significant': g1['boot_p_value'] < alpha,
        }

        g2 = self.glmm2_results
        for (variable, variable_label) in (('gamma_c', 'number_effect'), ('is_variant', 'nec_variant_effect')):
            df_ne = g2.xs(variable, level='metric')

            d |= {
                f'{variable_label}_log_or': df_ne['boot_median_log'],
                f'{variable_label}_or': np.exp(df_ne['boot_median_log']),
                f'{variable_label}_p_value': df_ne['boot_p_value'],
                f'{variable_label}_significant': df_ne['boot_p_value'] < alpha,
                f'{variable_label}_agreement': df_ne['agreement'],
            }

        df = pd.DataFrame(d).transpose()
        if self.models:
            df = df[[col for col in df.columns if col in self.models]]
        return df


class MultiPromptResult:
    """Helper to aggregate multiple PromptResult objects for comparison.

    The `summary` attribute contains a multi-indexed DataFrame suitable for
    the plotting helpers in `plotting_utils`.
    """
    def __init__(self, prompt_results: dict[str, PromptResult], save_prefix=None):
        self.prompt_results = prompt_results
        self.summary = pd.concat(
            [r.summary() for r in prompt_results.values()],
            keys=[r.short_label for r in prompt_results.values()],
            names=['prompt', 'quantity']
        )
        self.save_prefix = save_prefix

    def plot_prompt_comparison(self, models: list[str] | None = None, **kwargs):
        fig = plot_prompt_comparison(
            self.summary,
            colours={r.short_label: r.colour.lighten(factor=0.3).value for r in self.prompt_results.values()},
            models=models,
            save_prefix=self.save_prefix,
            **kwargs
        )
        return fig

    def plot_prompt_acc_evolution(self, models: list[str] | None = None, **kwargs):
        fig = plot_prompt_acc_evolution(
            self.summary,
            colours={r.short_label: r.colour.value for r in self.prompt_results.values()},
            models=models,
            save_prefix=self.save_prefix,
            **kwargs
        )
        return fig

    def glmm2_to_latex(self, v="number_effect", models: list[str] | None = None, position='H'):
        def get_q(name):
            s = self.summary
            if models:
                s = s[models]
            return s.xs(name, level='quantity').T

        odds_ratios = get_q(f'{v}_or')
        p_values = get_q(f'{v}_p_value')
        agreement = get_q(f'{v}_agreement')

        p_values_corrected = p_values.apply(
            lambda col: multipletests(col, alpha=0.05, method='holm')[1],
            axis=1,
            result_type='broadcast'
        )

        # Create an empty DataFrame with the same shape
        df_combined = pd.DataFrame(index=odds_ratios.index, columns=odds_ratios.columns)

        # Iterate and apply formatting
        for col in df_combined.columns:
            for idx in df_combined.index:
                df_combined.at[idx, col] = self.format_glmm2_cell(
                    odds_ratios.at[idx, col],
                    agreement.at[idx, col],
                    p_values.at[idx, col],
                    p_values_corrected.at[idx, col]
                )

        # Calculate column format: 'l' for index, 'c' for each column
        col_format = "l" + "c" * len(df_combined.columns)

        caption = ("Odds ratios and significance of the number effect across models and prompt formats. "
                   "Formatted as: Odds Ratio \\\\ Raw $p$ / Corrected $p$.")
        if not agreement.all().all():
            caption += r" The $\ddagger$ symbol marks cases where Wald-derived statistical significance status does not match the bootstrap one."

        # Export to LaTeX. escape=False is crucial here so pandas doesn't break our LaTeX tags.
        latex_table = df_combined.to_latex(
            escape=False,
            column_format=col_format,
            caption=caption,
            label=f"tab:glmm2_{v}_odds",
            position=position
        )

        print(latex_table)

    @staticmethod
    def format_glmm2_cell(or_val, agreement, p_raw, p_corr):
        """Combine OR and p-values into a LaTeX makecell string.

        The returned string contains the odds ratio on the first line and the
        raw/corrected p-values on the second line.
        """

        pf = format_p_value(3, use_delta=True)

        or_str = f"{or_val:.2f}"
        p_raw_str = pf(p_raw)
        if not agreement:
            p_raw_str += r" $\ddagger$"  # add a dagger symbol for non-convergent fits

        p_corr_str = pf(p_corr)

        # \\\\ tells makecell to break the line inside the table cell
        return f"\\makecell{{{or_str} \\\\ {p_raw_str} / {p_corr_str}}}"
