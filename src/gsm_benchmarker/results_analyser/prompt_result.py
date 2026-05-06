from pathlib import Path
import pandas as pd
from IPython.display import display
from dataclasses import dataclass
import numpy as np
from functools import cached_property

from gsm_benchmarker.results_analyser import MultiVariantMultiModelResultsAnalyser
from gsm_benchmarker.results_analyser.prompt_effect_analyser import PromptEffectAnalyser
from gsm_benchmarker.results_analyser.plotting_utils import plot_glmm, plot_acc_change_distribution, Colour
from gsm_benchmarker.results_analyser.utils import pandas_to_latex, correct_p_values


@dataclass
class PromptResult:
    path: str | Path
    colour: Colour
    full_label: str
    short_label: str | None = None
    models: list[str] | None = None
    metric: str | None = None
    notebook: bool = False
    save_dest: Path | None = None
    mres: MultiVariantMultiModelResultsAnalyser | None = None
    baseline: MultiVariantMultiModelResultsAnalyser | None = None
    pea: PromptEffectAnalyser | None = None
    use_difficulty: bool = True

    def __post_init__(self):
        if self.mres is None:
            self.mres = MultiVariantMultiModelResultsAnalyser(self.path)

        if self.baseline is not None and self.pea is None:
            assert self.mres is not None
            self.pea = PromptEffectAnalyser(self.baseline, self.mres, self.full_label)

        if self.short_label is None:
            self.short_label = self.full_label.split(' ')[0]

    @cached_property
    def variant_effect(self) -> pd.DataFrame:
        assert self.mres is not None
        return self.mres.analyse_variant_effect(
            variant='main', metric=self.metric, models=self.models, use_difficulty=self.use_difficulty)

    def variant_effect_to_latex(self, alpha=0.05, projected_alpha: float | None = None, model_order: list[str] | None = None):
        df = self.variant_effect.copy()

        if self.models is not None:
            df = df[df.index.isin(self.models)]

        if model_order is not None:
            df = df.sort_index(
                key=lambda c: c.map({model: index for index, model in enumerate(model_order)})
            )

        df['odds_ratio'] = np.exp(df['estimate']).round(2)

        # Calculate the 95% CI bounds for the log-odds, then exponentiate them
        # The z-score for a 95% confidence interval is approx 1.96
        df['or_ci_lower'] = np.exp(df['estimate'] - (1.96 * df['std_err'])).round(2)
        df['or_ci_upper'] = np.exp(df['estimate'] + (1.96 * df['std_err'])).round(2)

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
            'GSM8K acc': df['GSM8K_acc'].apply(fmt(1)),
            'main acc': df['main_acc'].apply(fmt(1)),
            r'$\Delta$ Acc': df['acc_diff'].apply(fmt(2)),
            'P value': df['p_value'].apply(fmt_p_val(3)),
            'Corrected P value': correct_p_values(df['p_value']).apply(fmt_p_val(3))
        }, index=df.index)
        df1.index.name = 'Model'

        df2 = pd.DataFrame({
            'Odds ratio': df['odds_ratio'].apply(fmt(2)),
            r'95\% CI': df.apply(lambda row: f"[{row['or_ci_lower']:.2f}, {row['or_ci_upper']:.2f}]", axis=1),
            'Z value': df['z_value'].apply(fmt(2)),
            'Std. error': df['std_err'].apply(fmt(2))
        }, index=df.index)
        df2.index.name = 'Model'

        df1_latex = pandas_to_latex(df1, label=f"tab:{self.short_label}-results", caption=f"Results of {self.full_label}", clean_header=False)
        df2_latex = pandas_to_latex(df2, label=f"tab:{self.short_label}-stats", caption=f"Additional statistics for results of {self.full_label}", clean_header=False)

        print(df1_latex)
        print(df2_latex)

    def _check_pea(self):
        if self.pea is None:
            raise ValueError(f"Prompt effect analysis not possible for baseline prompt ({self.full_label})")
        return self.pea

    def _make_prompt_effect_df(self, variant):
        return self._check_pea().analyse_accuracy_change_significance(
            variant=variant, models=self.models, metric=self.metric, use_difficulty=self.use_difficulty)

    @cached_property
    def prompt_effect_main(self) -> pd.DataFrame:
        return self._make_prompt_effect_df('main')

    @cached_property
    def prompt_effect_gsm8k(self) -> pd.DataFrame:
        return self._make_prompt_effect_df('GSM8K')

    def _make_number_effect_df(self, variant):
        assert self.mres is not None
        return  self.mres.analyse_number_effect(variant=variant, metric=self.metric, models=self.models)

    @cached_property
    def number_effect_main(self) -> pd.DataFrame:
        return self._make_number_effect_df('main')

    @cached_property
    def number_effect_gsm8k(self) -> pd.DataFrame:
        return self._make_number_effect_df('GSM8K')

    def display_plots(self, *figs):
        if self.notebook:
            for fig in figs:
                display(fig)
        return figs

    def plot_variant_effect(self, **kwargs):
        figs = plot_glmm(
            self.variant_effect,
            'acc_diff',
            "Symbolic performance delta, pp",
            bar_colour=self.colour.value,
            save_prefix=self.save_dest/self.short_label if self.save_dest is not None else None,
            **kwargs
        )

        return self.display_plots(*figs)

    def plot_prompt_effect(self, **kwargs):
        figs = plot_glmm(
            self.prompt_effect_main,
            'acc_diff',
            "Prompt performance delta, pp",
            bar_colour=self.colour.value,
            save_prefix=self.save_dest/(self.short_label + "_pe") if self.save_dest is not None else None,
            **kwargs
        )

        return self.display_plots(*figs)

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

        return self.display_plots(fig)

    def get_significant_models(self, alpha: float, drop_only: bool = False):
        df = self.variant_effect
        if drop_only:
            df = df[df.acc_diff < 0]
        models = df[df.p_value < alpha].sort_values('estimate', ascending=True).index.tolist()
        return models

    def summary(self, alpha: float = 0.05):
        assert self.mres is not None

        d = {
            'GSM8K_acc': self.mres.variants['GSM8K'].get_accuracies_per_model(metric=self.metric),
            'main_acc': self.mres.variants['main'].get_accuracies_per_model(metric=self.metric),
            'delta_symb': self.variant_effect['acc_diff'],
            'delta_symb_p_value': self.variant_effect['p_value'],
            'delta_symb_significant': self.variant_effect['p_value'] < alpha
        }

        for label in ('main', 'gsm8k'):
            # add 'number effect' - influence of bigger numbers on odds of getting a correct answer
            df_ne = getattr(self, f"number_effect_{label}")
            d |= {
                f'number_effect_{label}': df_ne['odds_change'],
                f'number_effect_{label}_p_value': df_ne['p_value'],
                f'number_effect_{label}_significant': df_ne['p_value'] < alpha
            }

            # add prompt effect data
            if self.baseline is None:
                continue
            df_pe = getattr(self, f"prompt_effect_{label}")
            d |= {
                f'delta_prompt_{label}': df_pe['acc_diff'],
                f'delta_prompt_{label}_p_value': df_pe['p_value'],
                f'delta_prompt_{label}_significant': df_pe['p_value'] < alpha
            }

        df = pd.DataFrame(d).transpose()
        if self.models:
            df = df[[col for col in df.columns if col in self.models]]
        return df
