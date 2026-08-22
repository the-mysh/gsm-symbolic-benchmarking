from functools import cached_property
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import logging

from gsm_benchmarker.scripts.bootstrap import make_names

logger = logging.getLogger(__name__)


class BootstrapResult:
    def __init__(self, data_path: Path | str, n_boot: int, glmm_id: str, alpha=0.05):
        self.data_path = Path(data_path)
        self.n_boot = n_boot

        output_filename, checkpoints_filename = make_names(n_boot, glmm_id)

        self.summary_df = self._load_summary(self.data_path / output_filename, alpha=alpha)

        try:
            self.full_results = pd.read_pickle(self.data_path / checkpoints_filename)
        except FileNotFoundError:
            logger.warning("Full results not available")

    def _load_summary(self, summary_path, alpha=0.05):

        summary_df = pd.read_pickle(summary_path)

        s_sig = summary_df['single_significant'] = summary_df['single_p_value'] < alpha
        s_width = summary_df['single_ci_width_log'] = summary_df['single_ci_upper_log'] - summary_df['single_ci_lower_log']
        s_est = summary_df['single_estimate']

        b_sig = summary_df[f'boot_significant'] = summary_df['boot_p_value'] < alpha

        summary_df[f'bias_log'] = summary_df[f'boot_mean_log'] - s_est

        summary_df[f'boot_ci_width_log'] = summary_df[f'boot_ci_upper_log'] - summary_df[f'boot_ci_lower_log']
        summary_df[f'width_ratio_log'] = summary_df[f'boot_ci_width_log'] / s_width

        summary_df[f'agreement'] = (s_sig == b_sig)

        return summary_df

    @cached_property
    def summary_numbers(self):
        return self._make_summary('boot_n_')

    def _make_summary(self, prefix: str, no_strip: bool = False):
        nc = len(prefix)

        cond = lambda s: s.startswith(prefix)
        trim = slice(nc, None)

        if no_strip:
            trim = slice(0, None)

        cols = [k for k in self.summary_df.columns if cond(k)]
        mapping = {k: k[trim] for k in cols}
        return self.summary_df[cols].rename(columns=mapping).rename(columns={'n': 'n_estimates'})

    @cached_property
    def summary_boot(self):
        return self._make_summary('boot_')

    def _get_single_not_stripped(self):
        return self._make_summary('single_', no_strip=True)

    @cached_property
    def summary_single(self):
        return self._make_summary('single_')

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
            'boot_ci_lower',
            'boot_ci_upper',
            'single_ci_lower',
            'single_ci_upper',
            'single_p_value',
            'single_nonconvergent'
        ]]

    def _get_variable_summary(self, variable: str | None = None):
        if variable is None:
            return self.summary_df
        return self.summary_df.xs(variable, level=1)

    def bias_check(self, variable: str | None = None):
        return self._get_variable_summary(variable).sort_values('bias', key=abs, ascending=False)[
            ['bias', 'boot_mean', 'single_estimate', 'single_ci_lower', 'single_ci_upper']]

    def ci_width_check(self, variable: str | None = None):
        df = self._get_variable_summary(variable)
        return df[~df.index.isin(self.get_nonconvergent_models(variable))][['width_ratio']].describe()

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
        s = self.summary_single
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
