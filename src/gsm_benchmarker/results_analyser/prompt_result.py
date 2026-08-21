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
from gsm_benchmarker.results_analyser.utils import pandas_to_latex, correct_p_values


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

    def __post_init__(self):
        if self.mres is None:
            self.mres = MultiVariantMultiModelResultsAnalyser(self.path)

        if self.baseline is not None and self.pea is None:
            assert self.mres is not None
            self.pea = PromptEffectAnalyser(self.baseline, self.mres, self.full_label)

        if self.short_label is None:
            self.short_label = self.full_label.split(' ')[0]

    def get_clean_data_object(self, full_label=None, short_label=None, colour=None, models=None):
        return PromptResult(
            path=self.path,
            colour=colour or self.colour,
            full_label=full_label or (self.full_label + " filtered"),
            short_label=short_label or (self.short_label + "_filtered"),
            models=models or self.models,
            metric=self.metric,
            save_dest=self.save_dest,
            mres=self.mres.get_clean_data_object(),
            baseline=self.baseline.get_clean_data_object() if self.baseline is not None else None,
            pea=None # pea created from the new mres and baseline
        )

    @cached_property
    def variant_effect(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute and cache the GLMM-based variant effect table for the prompt.

        The table contains coefficient estimates, p-values and derived odds
        ratios per model for the 'main' variant vs baseline.
        """
        return self.mres.analyse_variant_effect(variant='main', metric=self.metric, models=self.models)

    def variant_effect_to_latex(self, alpha=0.05, projected_alpha: float | None = None,
                                model_order: list[str] | None = None, position: str = "H"):
        df = self.variant_effect[0].copy()
        diagnostics_df = self.variant_effect[1]

        df['odds_ratio'] = np.exp(df['estimate']).round(2)
        df['convergent'] = ~diagnostics_df['convergence_messages'].astype(bool)

        # Calculate the 95% CI bounds for the log-odds, then exponentiate them
        # The z-score for a 95% confidence interval is approx 1.96
        df['or_ci_lower'] = np.exp(df['estimate'] - (1.96 * df['std_err'])).round(2)
        df['or_ci_upper'] = np.exp(df['estimate'] + (1.96 * df['std_err'])).round(2)

        # random effect variance and std. dev. of random effect
        df['ranef_variance'] = diagnostics_df['ranef_variance']
        df['ranef_sd'] = diagnostics_df['ranef_sd']

        if self.models is not None:
            df = df[df.index.isin(self.models)]

        if model_order is not None:
            df = df.sort_index(
                key=lambda c: c.map({model: index for index, model in enumerate(model_order)})
            )

        def fmt(precision=3):
            th = 10**(-precision)
            def wrapper(v):
                if not precision:
                    return str(round(v))
                if abs(v) < th:
                    return f"< {th:.{precision}f}"
                return f"{v:.{precision}f}"
            return wrapper

        def fmt_p_val(precision=3):
            str_fmt = fmt(precision=precision)
            a = projected_alpha if projected_alpha is not None else alpha
            def wrapper(v):
                v_formatted = str_fmt(v)
                if v < a:
                    return r"\textbf{" + v_formatted + "}"
                return v_formatted
            return wrapper

        df1 = pd.DataFrame({
            'GSM-Base acc': df['GSM8K_acc'].apply(fmt(1)),
            'GSM-Variants acc': df['main_acc'].apply(fmt(1)),
            r'$\Delta$ Acc': df['acc_diff'].apply(fmt(2)),
            'P value': df['p_value'].apply(fmt_p_val(3)),
            'Corrected P value': correct_p_values(df['p_value']).apply(fmt_p_val(3))
        }, index=df.index)
        df1.index.name = 'Model'


        df2 = pd.DataFrame({
            'Odds ratio': df.apply(lambda row: f"{row['odds_ratio']:.2f}{' $\\dagger$' if not row['convergent'] else ''}", axis=1),
            r'95\% CI': df.apply(lambda row: f"[{row['or_ci_lower']:.2f}, {row['or_ci_upper']:.2f}]", axis=1),
            'Z value': df['z_value'].apply(fmt(2)),
            'Std. error': df['std_err'].apply(fmt(2)),
            'Rand. eff. variance $\pm$ std. dev': df.apply(lambda row: f"{row['ranef_variance']:.2f} $\pm$ {row['ranef_sd']:.2f}", axis=1)
        }, index=df.index)
        df2.index.name = 'Model'

        caption1 = f"Results of {self.full_label}"
        df1_latex = pandas_to_latex(df1, label=f"tab:{self.short_label}-results", caption=caption1,
                                    clean_header=False, position=position)

        caption2 = f"Additional statistics for results of {self.full_label}."
        if not df['convergent'].all():
            caption2 += " Cases of non-convergent fits marked with $\\dagger$."
        df2_latex = pandas_to_latex(df2, label=f"tab:{self.short_label}-stats", caption=caption2,
                                    clean_header=False, position=position)

        print(df1_latex)
        print(df2_latex)

    def _check_pea(self):
        if self.pea is None:
            raise ValueError(f"Prompt effect analysis not possible for baseline prompt ({self.full_label})")
        return self.pea

    @cached_property
    def number_effect(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.mres.analyse_number_effect('main', metric=self.metric, models=self.models)

    def plot_variant_effect(self, **kwargs):
        """Produce GLMM visualisations for the variant effect of this prompt."""
        figs = plot_glmm(
            *self.variant_effect,
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
        df = self.variant_effect[0]
        if drop_only:
            df = df[df.acc_diff < 0]
        models = df[df.p_value < alpha].sort_values('estimate', ascending=True).index.tolist()
        return models

    def summary(self, alpha: float = 0.05):

        variant_effect_df, variant_effect_diagnostics_df = self.variant_effect

        d = {
            'GSM8K_acc': self.mres.variants['GSM8K'].get_accuracies_per_model(metric=self.metric),
            'main_acc': self.mres.variants['main'].get_accuracies_per_model(metric=self.metric),
            'delta_symb_acc_diff': variant_effect_df['acc_diff'],
            'delta_symb_log_or': variant_effect_df['estimate'],
            'delta_symb_or': variant_effect_df['odds_ratio'],
            'delta_symb_p_value': variant_effect_df['p_value'],
            'delta_symb_significant': variant_effect_df['p_value'] < alpha,
            'delta_symb_converged': ~variant_effect_diagnostics_df['convergence_messages'].astype(bool),
            'delta_symb_singular': variant_effect_diagnostics_df['is_singular'],
        }

        number_effect_df, number_effect_diagnostics_df = self.number_effect
        for (variable, variable_label) in (('sum_logs_c', 'number_effect'), ('is_variant', 'delta_symb_ne')):
            df_ne = number_effect_df.xs(variable, level='variable')

            d |= {
                f'{variable_label}_log_or': df_ne['estimate'],
                f'{variable_label}_or': df_ne['odds_ratio'],
                f'{variable_label}_p_value': df_ne['p_value'],
                f'{variable_label}_significant': df_ne['p_value'] < alpha,
                f'{variable_label}_converged': ~number_effect_diagnostics_df['convergence_messages'].astype(bool),
                f'{variable_label}_singular': number_effect_diagnostics_df['is_singular'],
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

    def number_effect_to_latex(self, v="number_effect", models: list[str] | None = None, position='H'):
        def get_q(name):
            return self.summary[models].xs(name, level='quantity').T

        odds_ratios = get_q(f'{v}_or')
        p_values = get_q(f'{v}_p_value')
        convergent = get_q(f'{v}_converged')

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
                df_combined.at[idx, col] = self.format_cell(
                    odds_ratios.at[idx, col],
                    convergent.at[idx, col],
                    p_values.at[idx, col],
                    p_values_corrected.at[idx, col]
                )

        # Calculate column format: 'l' for index, 'c' for each column
        col_format = "l" + "c" * len(df_combined.columns)

        caption = ("Odds ratios and significance of the number effect across models and prompt formats. "
                   "Formatted as: Odds Ratio \\\\ Raw $p$ / Corrected $p$.")
        if not convergent.all().all():
            caption += " Cases of non-convergent fits marked with $\\dagger$."

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
    def format_p_value(p):
        """Format a single p-value for LaTeX output.

        Values under 0.001 are shown as a delta symbol; values under 0.05 are
        additionally made bold.
        """
        # Check if p-value is less than 0.001
        if p < 0.001:
            # Use math mode for the less-than sign in LaTeX
            base_str = r"$< \delta$"
        else:
            # Format to 3 decimal places
            base_str = f"{p:.3f}"

        # Apply bold if under 0.05 threshold
        if p < 0.05:
            return f"\\textbf{{{base_str}}}"

        return base_str

    @staticmethod
    def format_cell(or_val, convergent, p_raw, p_corr):
        """Combine OR and p-values into a LaTeX makecell string.

        The returned string contains the odds ratio on the first line and the
        raw/corrected p-values on the second line.
        """
        or_str = f"{or_val:.2f}"
        if not convergent:
            or_str += " $\\dagger$"  # add a dagger symbol for non-convergent fits
        p_raw_str = MultiPromptResult.format_p_value(p_raw)
        p_corr_str = MultiPromptResult.format_p_value(p_corr)

        # \\\\ tells makecell to break the line inside the table cell
        return f"\\makecell{{{or_str} \\\\ {p_raw_str} / {p_corr_str}}}"
