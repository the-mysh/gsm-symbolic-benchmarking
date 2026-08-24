from functools import cached_property
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import logging

from gsm_benchmarker.scripts.utils import make_bootstrap_names

logger = logging.getLogger(__name__)


class BootstrapResult:
    def __init__(self, data_path: Path | str, n_boot: int, glmm_id: str, alpha=0.05):
        self.data_path = Path(data_path)
        self.n_boot = n_boot

        bootstrap_filename, wald_filename, checkpoints_filename = make_bootstrap_names(n_boot, glmm_id)
        self.boot_df = pd.read_pickle(self.data_path / bootstrap_filename).sort_index()
        self.wald_df = pd.read_pickle(self.data_path / wald_filename).sort_index()

        self.comparison_df = self._combine_results(alpha=alpha)

        try:
            self.full_results = pd.read_pickle(self.data_path / checkpoints_filename)
        except FileNotFoundError:
            logger.warning("Full results not available")

    def _combine_results(self, alpha=0.05):
        # inconsistent data saving error patch
        self.boot_df.index.names = self.wald_df.index.names

        self.wald_df['significant'] = self.wald_df['p_value'] < alpha
        self.boot_df['significant'] = self.boot_df['p_value'] < alpha

        self.wald_df['ci_width_log'] = self.wald_df['ci_upper_log'] - self.wald_df['ci_lower_log']
        self.boot_df['ci_width_log'] = self.boot_df['ci_upper_log'] - self.boot_df['ci_lower_log']

        comparison_df = pd.DataFrame({
            'agreement': self.boot_df['significant'] == self.wald_df['significant'],
            'width_ratio_log': self.boot_df['ci_width_log'] / self.wald_df['ci_width_log'],
            'bias_log': self.boot_df['median_log'] - self.wald_df['estimate']
        }, index=self.boot_df.index)

        return comparison_df

    @cached_property
    def summary_df(self):
        return pd.concat([
            self.boot_df.rename(columns={k: f"boot_{k}" for k in self.boot_df.columns}),
            self.wald_df.rename(columns={k: f"wald_{k}" for k in self.wald_df.columns}),
            self.comparison_df
        ], axis=1)

    @cached_property
    def boot_numbers(self):
        cols = [k for k in self.boot_df.columns if k.startswith('n_')]
        return self.boot_df[cols].rename(columns={k: k[2:] for k in cols})

    def _get_one_field_from_full_results(self, field: str):
        return {k: v[field] for k, v in self.full_results.items()}

    @cached_property
    def estimates(self):
        return self._get_one_field_from_full_results('estimates')

    def disagreements_check(self, variable: str | None = None):
        summary_df = self._get_variable_summary(variable)
        agreement = summary_df.agreement
        print(f"Agreement: {agreement.sum()} / {len(agreement)} models")

        # show any disagreements directly
        return summary_df[~agreement][[
            'boot_ci_lower_log',
            'boot_ci_upper_log',
            'wald_ci_lower_log',
            'wald_ci_upper_log',
            'wald_p_value',
            'wald_nonconvergent'
        ]]

    def _get_variable_summary(self, variable: str | None = None):
        if variable is None:
            return self.summary_df
        return self.summary_df.xs(variable, level=1)

    def bias_check(self, variable: str | None = None):
        return self._get_variable_summary(variable).sort_values('bias_log', key=abs, ascending=False)[
            ['bias_log', 'boot_mean_log', 'boot_median_log', 'wald_estimate', 'wald_ci_lower_log', 'wald_ci_upper_log']]

    def ci_width_check(self, variable: str | None = None):
        df = self._get_variable_summary(variable)
        return df[~df.index.isin(self.get_nonconvergent_models(variable))][['width_ratio_log']].describe()

    def skew_check(self, variable: str, threshold: float = 0.5):

        skews = {}
        for model_name, res in self.estimates.items():
            skew = res[variable].skew()
            if abs(skew) > 0.5:  # flag anything notably skewed
                skews[model_name] = skew

        if skews:
            print(f"Models with absolute skew > {threshold}:")
            for model_name, skew in skews.items():
                print("\t", model_name, skew)
        else:
            print(f"No models with absolute skew > {threshold}")

    def get_nonconvergent_models(self, variable: str | None = None):
        s = self.wald_df
        if variable is not None:
            s = s.xs(variable, level=1)
        return s[s.nonconvergent].index.tolist()

    def plot_nonagreeing_estimates(self, variable):
        summary = self._get_variable_summary(variable)
        agreement = summary.agreement
        estimates = self.estimates

        nonagreeing_models = summary[~agreement].index.tolist()

        for m in self.get_nonconvergent_models(variable):
            if m in nonagreeing_models:
                nonagreeing_models.remove(m)

        for model_name in nonagreeing_models:
            m_est = estimates[model_name][variable]
            plt.hist(m_est, bins=40)
            plt.axvline(0, color='red', linestyle='--')
            plt.title(model_name)
            plt.show()
            print(f"{model_name} / '{variable}': skew={m_est.skew():.2f}, "
                  f"% of resamples > 0: {(m_est > 0).mean()*100:.1f}%")
