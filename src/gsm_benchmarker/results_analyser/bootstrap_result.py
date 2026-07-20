from functools import cached_property
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import logging


logger = logging.getLogger(__name__)


def pick_version(attr_fmt: str = "{}_combined"):
    def decorator(func):
        def wrapper(inst, inclusive: bool = False, **kwargs):
            attr_name = attr_fmt.format('inclusive' if inclusive else 'clean')
            bsc = getattr(inst, attr_name)
            return func(inst, bsc, **kwargs)
        return wrapper
    return decorator


class BootstrapResult:
    def __init__(self, data_path: Path | str, n_boot: int):
        self.data_path = Path(data_path)
        self.n_boot = n_boot

        self.summary_df = self._load_summary(self.data_path / f'boot{n_boot}.csv')

        try:
            self.full_results = pd.read_pickle(self.data_path / f'boot{n_boot}_checkpoints.pkl')
        except FileNotFoundError:
            logger.warning("Full results not available")

    def _load_summary(self, summary_path):

        summary_df = pd.read_csv(summary_path, index_col=0)

        s_sig = summary_df['single_significant'] = summary_df['single_p_value'] < 0.05  # bring in your Table 2/3 p-values per model
        s_width = summary_df['single_ci_width'] = summary_df['single_ci_upper'] - summary_df['single_ci_lower']
        s_est = summary_df['single_estimate']

        for label in ('clean', 'inclusive'):
            ci_contains_zero = (summary_df[f'ci_lower_{label}'] <= 0) & (summary_df[f'ci_upper_{label}'] >= 0)
            b_sig = summary_df[f'significant_{label}'] = ~ci_contains_zero

            summary_df[f'bias_{label}'] = summary_df[f'boot_mean_{label}'] - s_est

            summary_df[f'ci_width_{label}'] = summary_df[f'ci_upper_{label}'] - summary_df[f'ci_lower_{label}']
            summary_df[f'width_ratio_{label}'] = summary_df[f'ci_width_{label}'] / s_width

            summary_df[f'agreement_{label}'] = (s_sig == b_sig)


        return summary_df

    @cached_property
    def summary_numbers(self):
        return self._stripped_summary('n_', is_prefix=True, no_strip=True)

    def _stripped_summary(self, label: str, is_prefix: bool = False, no_strip: bool = False):
        nc = len(label)

        if is_prefix:
            cond = lambda s: s.startswith(label)
            trim = slice(nc, None)
        else:
            cond = lambda s: s.endswith(label)
            trim = slice(0, -nc)

        if no_strip:
            trim = slice(0, None)

        cols = [k for k in self.summary_df.columns if cond(k)]
        mapping = {k: k[trim] for k in cols}
        return self.summary_df[cols].rename(columns=mapping).rename(columns={'n': 'n_estimates'})

    @cached_property
    def summary_clean(self):
        return self._stripped_summary('_clean')

    def _get_single_not_stripped(self):
        return self._stripped_summary('single_', is_prefix=True, no_strip=True)

    @cached_property
    def clean_combined(self):
        return self.summary_clean.join(self._get_single_not_stripped())

    @cached_property
    def summary_inclusive(self):
        return self._stripped_summary('_inclusive')

    @cached_property
    def inclusive_combined(self):
        return self.summary_inclusive.join(self._get_single_not_stripped())

    @cached_property
    def summary_single(self):
        return self._stripped_summary('single_', is_prefix=True)

    def _get_one_field_from_full_results(self, field: str):
        return {k: v[field] for k, v in self.full_results.items()}

    @cached_property
    def clean_estimates(self):
        return self._get_one_field_from_full_results('estimates_clean')

    @cached_property
    def inclusive_estimates(self):
        return self._get_one_field_from_full_results('estimates_inclusive')

    @pick_version()
    def disagreements_check(self, bsc):
        agreement = bsc.agreement
        print(f"Agreement: {agreement.sum()} / {len(agreement)} models")

        return bsc[~agreement][['ci_lower', 'ci_upper', 'single_ci_lower', 'single_ci_upper', 'single_p_value', 'single_nonconvergent']]  # show any disagreements directly

    @pick_version()
    def bias_check(self, bsc):
        return bsc.sort_values('bias', key=abs, ascending=False)[[f'bias', 'boot_mean', 'single_estimate', 'single_ci_lower', 'single_ci_upper']]

    @pick_version()
    def ci_width_check(self, bsc):
        return bsc[~bsc.index.isin(self.nonconvergent_models)][['width_ratio']].describe()

    @pick_version("{}_estimates")
    def skew_check(self, estimates, threshold: float = 0.5):

        skews = {}
        for model_name, res in estimates.items():
            skew = pd.Series(res).skew()
            if abs(skew) > 0.5:  # flag anything notably skewed
                skews[model_name] = skew

        if skews:
            print(f"Models with absolute skew > {threshold}:")
            for model_name, skew in skews.items():
                print("\t", model_name, skew)
        else:
            print(f"No models with absolute skew > {threshold}")

    @property
    def nonconvergent_models(self):
        return self.summary_single[self.summary_single.nonconvergent].index.tolist()

    def plot_nonagreeing_estimates(self, inclusive: bool = False):
        if inclusive:
            agreement = self.inclusive_combined.agreement
            estimates = self.inclusive_estimates
        else:
            agreement = self.clean_combined.agreement
            estimates = self.clean_estimates

        nonagreeing_models = self.summary_df[~agreement].index.tolist()

        for m in self.nonconvergent_models:
            if m in nonagreeing_models:
                nonagreeing_models.remove(m)

        for model_name in nonagreeing_models:
            m_est = estimates[model_name]
            plt.hist(m_est, bins=40)
            plt.axvline(0, color='red', linestyle='--')
            plt.title(model_name)
            plt.show()
            print(f"{model_name}: skew={pd.Series(m_est).skew():.2f}, "
                  f"% of resamples > 0: {(m_est > 0).mean()*100:.1f}%")
