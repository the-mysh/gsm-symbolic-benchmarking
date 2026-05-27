"""GLMM helpers that depend on R (rpy2 / pymer4).

This module contains the GLMM runner wrapper and related exceptions. It
isolates R-dependent code so other parts of the package can import
non-R utilities without requiring rpy2 to be available.
"""

import logging
from typing import TYPE_CHECKING
import pandas as pd
import numpy as np
import re

from rpy2.rinterface_lib.embedded import RRuntimeError
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

# Set pandas converter as the global default for rpy2
ro.conversion.set_conversion(pandas2ri.converter + ro.default_converter)

from pymer4.models import glmer  # needs to go after the converter setting

if TYPE_CHECKING:
    from gsm_benchmarker.results_analyser.multi_model import MultiModelResultsAnalyser

logger = logging.getLogger(__name__)


class GLMMFitError(RuntimeError):
    """Raised when fitting a GLMM model fails at the R level.

    The wrapper code intentionally captures R runtime errors and raises this
    Python exception in cases where no R model object is available.
    """
    pass


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
        self._labels = list(set(set(re.findall(r"[\w\:]+", fixed_effects_term))))

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

        return coefs_df

    def prep_df_with_bool_labels(self, metric: str, ras: dict[int, "MultiModelResultsAnalyser"]) -> pd.DataFrame:
        """Prepare a combined DataFrame for GLMM fitting from two analysers.

        The `ras` mapping should map integer labels (e.g. 0, 1) to
        MultiModelResultsAnalyser instances. The method extracts per-example
        metric values and creates columns required by the GLMM (is_correct and
        the fixed effect flag).
        """

        if len(self._labels) > 1:
            raise RuntimeError("Cannot automatically prep df with multiple fixed effects")

        def _prep(label_value: bool, ra: "MultiModelResultsAnalyser"):
            res = ra.full_data
            res = res[['model', 'id', metric]][:]
            res[self._fixed_effects_term] = [label_value] * len(res)
            res['is_correct'] = res[metric].astype(int)
            res = res.drop(metric, axis=1)
            return res

        df = pd.concat([_prep(key, value) for key, value in ras.items()]).reset_index(drop=True)

        return df

    def run(self, df: pd.DataFrame, models: list[str] | None = None, simplify=False):
        """Run per-model GLMM fits on data grouped by 'model'.

        Returns a DataFrame with coefficient estimates and statistics for each
        model. If `simplify` is True and there is only one variable in the
        results, the returned DataFrame is simplified to have model as index.
        """

        glmm_results = {}

        for model_name, group_df in df.groupby('model'):
            if models is not None and model_name not in models:
                continue

            for label in self._labels:
                try:
                    group_df = group_df.dropna(subset=[label])  # make sure there are no NaNs
                except KeyError:
                    pass

            group_df = group_df[[c for c in group_df.columns if c != 'model']]

            try:
                coefs_df = self.fit_df(group_df)
            except GLMMFitError as err:
                logger.warning(f"{model_name}: {err}")
                res = {'estimate': np.nan, 'p_value': 1, 'std_err': np.nan, 'z_value': np.nan}
                coefs_df = pd.DataFrame(len(self._labels) * [res], index=self._labels)
            else:
                coefs_df.rename(columns={'Estimate': 'estimate', 'Pr(>|z|)': 'p_value', 'Std. Error': 'std_err', 'z value': 'z_value' }, inplace=True)
                coefs_df.drop('(Intercept)', axis=0, inplace=True)

            glmm_results[model_name] = coefs_df

        glmm_results_df = pd.concat(glmm_results.values(), keys=glmm_results.keys(), names=['model', 'variable'])
        if simplify and len(glmm_results_df.index.get_level_values('variable').unique()) < 2:
            glmm_results_df = glmm_results_df.reset_index().drop('variable', axis=1).set_index('model')

        if models is not None:
            models_with_results = glmm_results_df.index.get_level_values('model').unique()
            for requested_model_name in models:
                if requested_model_name not in models_with_results:
                    logger.warning(f"No data for model {requested_model_name}")

        return glmm_results_df

