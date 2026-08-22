"""GLMM helpers that depend on R (rpy2 / pymer4).

This module contains the GLMM runner wrapper and related exceptions. It
isolates R-dependent code so other parts of the package can import
non-R utilities without requiring rpy2 to be available.
"""

import logging
from typing import NamedTuple, Any
import pandas as pd
import numpy as np
import re
import time
from pathlib import Path
from tqdm import trange
import pickle

from rpy2.rinterface_lib.embedded import RRuntimeError
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

# Set pandas converter as the global default for rpy2
ro.conversion.set_conversion(pandas2ri.converter + ro.default_converter)

from pymer4.models import glmer  # needs to go after the converter setting


logger = logging.getLogger(__name__)


class GLMMFitError(RuntimeError):
    """Raised when fitting a GLMM model fails at the R level.

    The wrapper code intentionally captures R runtime errors and raises this
    Python exception in cases where no R model object is available.
    """
    pass


class FitResult(NamedTuple):
    coefs_df: pd.DataFrame
    ranef_var_df: pd.DataFrame
    is_singular: bool
    convergence_messages: str


class BootstrapFitResult(NamedTuple):
    clean_estimates: pd.DataFrame
    singular_estimates: pd.DataFrame
    n_failed: int
    n_nonconverged: int
    n_singular: int
    n_clean: int


class GLMMRunner:
    """Helper for fitting Generalised Linear Mixed Models (GLMMs) via R.

    This class builds a glmer formula from a provided fixed effects term and
    an optional random-effects term. It exposes convenience methods to prepare
    data and run per-model fits, returning coefficients and p-values in a
    pandas.DataFrame.
    """

    def __init__(self, fixed_effects_term: str, random_effects_term: str = "(1 | id)"):
        self._formula = f'is_correct ~ {fixed_effects_term} + {random_effects_term}'
        self._fixed_effects_term = fixed_effects_term
        self._fixed_effects_labels = list(set(set(re.findall(r"[\w\:]+", fixed_effects_term))))

    @property
    def fixed_effect_labels(self):
        return self._fixed_effects_labels[:]  # return copy

    def fit_df(self, df: pd.DataFrame):
        """Fit a GLMM on a prepared DataFrame and return coefficient table.

        The DataFrame must be suitable for passing directly to pymer4.glmer
        (i.e. contains the response and predictor columns used in the
        formula). Returns the coefficients summary as a pandas.DataFrame.
        """

        glmm_model = glmer(
            self._formula,
            data=df,
            family='binomial'
        )

        try:
            glmm_model.fit(verbose=False)  # fitting works, only getting stats fails
        except RRuntimeError as err:
            if glmm_model.r_model is None:
                raise GLMMFitError(f"GLMM fitting failed: {err}")

        # Assign the model to an R variable first
        ro.globalenv['r_model'] = glmm_model.r_model

        # Then extract coefficients as a DataFrame
        with localconverter(ro.default_converter + pandas2ri.converter):
            coefs_df = ro.r('as.data.frame(coef(summary(r_model)))')
            ranef_var_df = ro.r('as.data.frame(VarCorr(r_model))')
            is_singular = ro.r('isSingular(r_model)')[0]  # True/False

            conv_check = ro.r('''
                msgs <- r_model@optinfo$conv$lme4$messages
                if (is.null(msgs)) "" else paste(msgs, collapse = "; ")
            ''')
            convergence_messages = str(conv_check[0])  # empty string means no convergence warnings

        return FitResult(
            coefs_df=coefs_df,
            ranef_var_df=ranef_var_df,
            is_singular=is_singular,
            convergence_messages=convergence_messages
        )

    def run(self, df: pd.DataFrame, models: list[str] | None = None, simplify=False):
        """Run per-model GLMM fits on data grouped by 'model'.

        Returns a tuple of (results_df, diagnostics_df). results_df has
        coefficient estimates and statistics for each model. If `simplify`
        is True and there is only one variable in the results, results_df
        is simplified to have model as index. diagnostics_df has one row
        per model with convergence status and random-effect variance info.
        """

        glmm_results = {}
        diagnostics_records = []

        for model_name, group_df in df.groupby('model'):
            if models is not None and model_name not in models:
                continue

            for label in self._fixed_effects_labels:
                try:
                    group_df = group_df.dropna(subset=[label])  # make sure there are no NaNs
                except KeyError:
                    pass

            group_df = group_df[[c for c in group_df.columns if c != 'model']]

            try:
                fit_result = self.fit_df(group_df)
            except GLMMFitError as err:
                logger.warning(f"{model_name}: {err}")
                res = {'estimate': np.nan, 'p_value': 1, 'std_err': np.nan, 'z_value': np.nan}
                coefs_df = pd.DataFrame(len(self._fixed_effects_labels) * [res], index=self._fixed_effects_labels)
                diagnostics_records.append({
                    'model': model_name,
                    'fit_failed': True,
                    'is_singular': np.nan,
                    'convergence_messages': str(err),
                    'ranef_variance': np.nan,
                    'ranef_sd': np.nan,
                })
            else:
                if fit_result.convergence_messages:
                    logger.warning(f"{model_name} - convergence messages: {fit_result.convergence_messages}")

                coefs_df = fit_result.coefs_df
                coefs_df.rename(
                    columns={
                        'Estimate': 'estimate',
                        'Pr(>|z|)': 'p_value',
                        'Std. Error': 'std_err',
                        'z value': 'z_value'},
                    inplace=True
                )
                coefs_df.drop('(Intercept)', axis=0, inplace=True)

                # ranef_var_df is expected to have one row per grouping factor (here, just 'Id')
                ranef_row = fit_result.ranef_var_df.iloc[0]
                diagnostics_records.append({
                    'model': model_name,
                    'fit_failed': False,
                    'is_singular': fit_result.is_singular,
                    'convergence_messages': fit_result.convergence_messages,
                    'ranef_variance': ranef_row.get('vcov', np.nan),
                    'ranef_sd': ranef_row.get('sdcor', np.nan),
                })

            glmm_results[model_name] = coefs_df

        glmm_results_df = pd.concat(glmm_results.values(), keys=glmm_results.keys(), names=['model', 'variable'])
        if simplify and len(glmm_results_df.index.get_level_values('variable').unique()) < 2:
            glmm_results_df = glmm_results_df.reset_index().drop('variable', axis=1).set_index('model')

        if models is not None:
            models_with_results = glmm_results_df.index.get_level_values('model').unique()
            for requested_model_name in models:
                if requested_model_name not in models_with_results:
                    logger.warning(f"No data for model {requested_model_name}")

        diagnostics_df = pd.DataFrame(diagnostics_records).set_index('model')
        self._add_odds_changes(glmm_results_df)

        return glmm_results_df, diagnostics_df

    @staticmethod
    def _add_odds_changes(df):
        odds_ratio = np.exp(df.estimate)
        df['odds_ratio'] = odds_ratio
        df['odds_change'] = odds_ratio - 1  # change in odds
        return df

    def bootstrap_fit_df(self, df: pd.DataFrame, n_boot: int = 1000, cluster_col: str = 'id', seed: int = None
                         ) -> BootstrapFitResult:
        rng = np.random.default_rng(seed)
        unique_ids = df[cluster_col].unique()
        n_clusters = len(unique_ids)

        clean_estimates = []        # converged, non-singular estimates
        singular_estimates = []     # converged but boundary/singular fit
        n_failed = 0                # number of 'hard' errors (GLMMFitError; not expecting any of these)
        n_nonconverged = 0          # number of estimates with convergence warnings - excluded from estimate pools


        for i in trange(n_boot):
            # randomly sample ids with replacement
            sampled_ids = rng.choice(unique_ids, size=n_clusters, replace=True)

            # rebuild resampled df, keeping all rows per sampled id, with duplicate ids relabeled
            # so lme4 doesn't collapse repeated cluster labels into one group
            resampled_frames = []
            for new_idx, orig_id in enumerate(sampled_ids):
                chunk = df[df[cluster_col] == orig_id].copy()
                chunk[cluster_col] = f"{orig_id}_{new_idx}"  # relabel to keep clusters distinct
                resampled_frames.append(chunk)
            boot_df = pd.concat(resampled_frames, ignore_index=True)

            try:
                fit_result = self.fit_df(boot_df)
                estimate = fit_result.coefs_df.loc[:, 'Estimate'].to_dict()
            except GLMMFitError:
                n_failed += 1
                continue

            if fit_result.convergence_messages:
                logger.debug("Non-convergent estimate - discarding")
                n_nonconverged += 1
                continue  # discard estimate - unreliable

            if fit_result.is_singular:
                logger.debug("Singular estimate - collecting separately")
                singular_estimates.append(estimate)
            else:
                clean_estimates.append(estimate)

        logger.debug(f"Bootstrap loop finished; encountered {n_failed} failed fits, "
                     f"{n_nonconverged} non-convergent estimates, and {len(singular_estimates)} singular estimates")

        return BootstrapFitResult(
            clean_estimates=pd.DataFrame(clean_estimates),
            singular_estimates=pd.DataFrame(singular_estimates),
            n_failed=n_failed,
            n_nonconverged=n_nonconverged,
            n_singular=len(singular_estimates),
            n_clean=len(clean_estimates)
        )

    def run_bootstrap(self, df: pd.DataFrame, models: list[str] | None = None, n_boot: int = 1000,
                      cluster_col: str = 'id', seed: int = 42, checkpoint_path: str | Path | None = None,
                      ignore_previous_checkpoints: bool = False) -> dict:

        n_models = len(models) if models is not None else df.model.unique().size
        logger.info(f"RUNNING FULL BOOTSTRAP for {n_models} models")

        checkpoint_file, results = self._load_bootstrap_checkpoint(checkpoint_path, ignore_previous_checkpoints)

        mi = 0

        for model_name, model_df in df.groupby('model'):
            if models is not None and model_name not in models:
                continue

            mi += 1

            if model_name in results:
                logger.debug(f"Skipping {model_name} (already in checkpoint).")
                continue

            logger.debug(f"Running bootstrap [{mi}/{n_models}]: {model_name} ({n_boot} resamples)...")
            start = time.time()

            model_result = self.bootstrap_fit_df(
                model_df, n_boot=n_boot, cluster_col=cluster_col, seed=seed
            )

            results[model_name] = self._summarise_model_bootstrap(
                model_name, model_result, n_boot, elapsed=time.time() - start)

            self._update_bootstrap_checkpoint(checkpoint_file, results)  # checkpoint after every model

        return results

    @staticmethod
    def _update_bootstrap_checkpoint(checkpoint_file: Path | None, results: dict) -> None:
        if checkpoint_file is None:
            return

        with open(checkpoint_file, 'wb') as f:
            logger.debug(f"Updating bootstrap checkpoint file at {checkpoint_file}")
            pickle.dump(results, f)

    @staticmethod
    def _load_bootstrap_checkpoint(checkpoint_path: str | Path | None, ignore_previous_checkpoints: bool = False
                                   ) -> tuple[Path | None, dict]:

        # Load existing results if resuming from a checkpoint

        results = {}

        if checkpoint_path is None:
            checkpoint_file = None
        else:
            checkpoint_file = Path(checkpoint_path).resolve()
            if not ignore_previous_checkpoints and checkpoint_file.exists():
                with open(checkpoint_file, 'rb') as f:
                    results = pickle.load(f)
                logger.info(f"Resuming from checkpoint: {len(results)} models already done.")

        return checkpoint_file, results

    def _summarise_model_bootstrap(self, model_name: str, model_result: BootstrapFitResult, n_boot: int,
                                   elapsed: float) -> dict:

        estimates = model_result.clean_estimates.loc[:, self._fixed_effects_labels]

        full_summary: dict[str, Any] = {'model': model_name, 'estimates': estimates}

        stats = {}
        for effect_label in self._fixed_effects_labels:
            effect_estimates = estimates.loc[:, effect_label]
            if len(effect_estimates):
                ci_lower, ci_upper = np.percentile(effect_estimates, [2.5, 97.5])
                boot_se = np.std(effect_estimates, ddof=1)
                boot_mean = np.mean(effect_estimates)
                boot_median = np.median(effect_estimates)
                boot_p_value = 2 * min((effect_estimates <= 0).mean(), (effect_estimates >= 0).mean())
            else:
                ci_lower = ci_upper = boot_se = boot_mean = boot_median = boot_p_value = np.nan

            stats[effect_label] = {
                "ci_upper_log": ci_upper,
                "ci_lower_log": ci_lower,
                "se_log": boot_se,
                "mean_log": boot_mean,
                "median_log": boot_median,
                "p_value": boot_p_value
            }

        full_summary['stats'] = pd.DataFrame(stats)

        full_summary['info'] = {
                'n_requested': n_boot,
                'n_failed': model_result.n_failed,
                'n_nonconverged': model_result.n_nonconverged,
                'n_singular': model_result.n_singular,
                'n_clean': model_result.n_clean,
                'elapsed_seconds': elapsed,
            }

        logger.debug(
            f"Done in {elapsed:.1f}s. "
            f"Obtained {len(model_result.clean_estimates)} clean estimates "
            f"and {len(model_result.singular_estimates)} singular estimates."
        )

        return full_summary

    @staticmethod
    def summarise_bootstrap_results(results: dict) -> pd.DataFrame:
        """
        Turn the results dict into a tidy summary DataFrame.
        """

        rows = {}
        for model_name, res in results.items():
            stats = res['stats']
            info = res['info']
            info_df = pd.DataFrame({v: info for v in stats.columns})
            rows[model_name] = stats.T.join(info_df.T)

        summary_df = pd.concat(rows)
        return summary_df

    @staticmethod
    def summarise_wald_results(estimates_df: pd.DataFrame, info_df: pd.DataFrame) -> pd.DataFrame:

        summary_df = estimates_df[['estimate', 'std_err', 'p_value']].copy()

        sest = estimates_df['estimate']
        sse = estimates_df['std_err']
        z_crit = 1.959964

        summary_df['ci_lower_log'] = sest - z_crit * sse
        summary_df['ci_upper_log'] = sest + z_crit * sse

        glmm_variables = estimates_df.index.get_level_values(1).unique()
        info_df = pd.concat({k: info_df for k in glmm_variables}).swaplevel()

        summary_df['singular'] = info_df['is_singular']
        summary_df['nonconvergent'] = info_df['convergence_messages'].astype(bool)
        summary_df['fit_failed'] = info_df['fit_failed']
        summary_df['ranef_variance'] = info_df['ranef_variance']
        summary_df['ranef_sd'] = info_df['ranef_sd']

        return summary_df
