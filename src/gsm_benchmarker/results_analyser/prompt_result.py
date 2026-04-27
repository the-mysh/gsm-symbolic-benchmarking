from pathlib import Path
import pandas as pd
from IPython.display import display
from dataclasses import dataclass

from gsm_benchmarker.results_analyser import MultiVariantMultiModelResultsAnalyser
from gsm_benchmarker.results_analyser.prompt_effect_analyser import PromptEffectAnalyser
from gsm_benchmarker.results_analyser.plotting_utils import plot_glmm, plot_acc_change_distribution, Colour, plot_prompt_format_comparison


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
    _variant_effect: pd.DataFrame | None = None
    _prompt_effect: pd.DataFrame | None = None

    def __post_init__(self):
        if self.mres is None:
            self.mres = MultiVariantMultiModelResultsAnalyser(self.path)

        if self.baseline is not None and self.pea is None:
            assert self.mres is not None
            self.pea = PromptEffectAnalyser(self.baseline, self.mres, self.full_label)

        if self.short_label is None:
            self.short_label = self.full_label.split(' ')[0]

    @property
    def variant_effect(self) -> pd.DataFrame:
        if self._variant_effect is None:
            assert self.mres is not None
            self._variant_effect = self.mres.analyse_variant_effect(variant='main', metric=self.metric, models=self.models)

        assert self._variant_effect is not None
        return self._variant_effect

    def _check_pea(self):
        if self.pea is None:
            raise ValueError(f"Prompt effect analysis not possible for baseline prompt ({self.full_label})")
        return self.pea


    @property
    def prompt_effect(self) -> pd.DataFrame:
        if self._prompt_effect is None:
            self._prompt_effect = self._check_pea().analyse_accuracy_change_significance(variant='main', models=self.models, metric=self.metric)

        assert self._prompt_effect is not None
        return self._prompt_effect


    def display_plots(self, *figs):
        if self.notebook:
            for fig in figs:
                display(fig)
        return figs

    def plot_variant_effect(self, **kwargs):
        figs = plot_glmm(
            self.variant_effect,
            'mean_diff',
            "Symbolic performance delta",
            bar_colour=self.colour.value,
            save_prefix=self.save_dest/self.short_label if self.save_dest is not None else None,
            **kwargs
        )

        return self.display_plots(*figs)

    def plot_prompt_effect(self, **kwargs):
        figs = plot_glmm(
            self.prompt_effect,
            'mean_diff',
            "Prompt performance delta",
            bar_colour=self.colour.lighten(),
            save_prefix=self.save_dest/(self.short_label + "_pe") if self.save_dest is not None else None,
            **kwargs
        )

        return self.display_plots(*figs)

    def plot_acc_change_dist(self, **kwargs):
        acc_change_raw = self._check_pea().get_accuracy_change(variant='main', metric=self.metric)

        fig = plot_acc_change_distribution(
            acc_change_raw,
            label="Prompt performance delta",
            models=self.models,
            color=self.colour.lighten(),
            save_prefix=self.save_dest/self.short_label if self.save_dest is not None else None,
            **kwargs
        )

        return self.display_plots(fig)

    def get_significant_models(self, alpha: float):
        df = self.variant_effect
        models = df[df.p_value < alpha].index.get_level_values('model').unique().tolist()
        return models

    def summary(self, alpha: float = 0.05):
        assert self.mres is not None

        d = {
            'GSM8K_acc': self.mres.variants['GSM8K'].get_accuracies_per_model(metric=self.metric),
            'main_acc': self.mres.variants['main'].get_accuracies_per_model(metric=self.metric),
            'delta_symb': self.variant_effect['mean_diff'],
            'delta_symb_p_value': self.variant_effect['p_value'],
            'delta_symb_significant': self.variant_effect['p_value'] < alpha
        }

        if self._prompt_effect is not None or self.baseline is not None:
            d |= {
                'delta_prompt': self.prompt_effect['mean_diff'],
                'delta_prompt_p_value': self.prompt_effect['p_value'],
                'delta_prompt_significant': self.prompt_effect['p_value'] < alpha
            }

        df = pd.DataFrame(d).transpose()
        if self.models:
            df = df[self.models]
        return df