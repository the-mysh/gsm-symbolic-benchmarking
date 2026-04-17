import pandas as pd
import logging
from typing import TYPE_CHECKING
import numpy as np

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
    pass


METRIC_LABELS = {'correct': 'standard', 'correct_strict': 'discounted'}

def do_for_metrics(func):
    def wrapper(*args, metric: str | None = None, **kwargs):

        if metric is None:
            res = {}
            for metric, metric_label in METRIC_LABELS.items():
                res[metric_label] = func(*args, metric=metric, **kwargs)

            df_results = pd.concat(res.values(), keys=res.keys(), names=('metric', 'model'))
            df_results = df_results.swaplevel().sort_index()
            return df_results
        else:
            return func(*args, metric=metric, **kwargs)
    return wrapper


class GLMMRunner:
    def __init__(self, label: str, question_difficulties: pd.DataFrame):
        self._formula = f'is_correct ~ {label} + difficulty + (1 | id)'
        self._label = label
        self._question_difficulties = question_difficulties

    def fit_df(self, df: pd.DataFrame):
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

    def prep_df(self, metric: str, ras: dict[int, "MultiModelResultsAnalyser"]) -> pd.DataFrame:
        def _prep(label_value: bool, ra: "MultiModelResultsAnalyser"):
            res = ra.full_data
            res = res[['model', 'id', metric]][:]
            res[self._label] = [label_value] * len(res)
            res['is_correct'] = res[metric].astype(int)
            res = res.drop(metric, axis=1)
            return res

        df = pd.concat([_prep(key, value) for key, value in ras.items()]).reset_index(drop=True)

        return df

    @do_for_metrics
    def run(self, ras: dict[int, "MultiModelResultsAnalyser"], metric: str, models: list[str] | None = None):
        df = self.prep_df(metric=metric, ras=ras)
        glmm_results = []

        for model_name, group_df in df.groupby('model'):
            if models is not None and model_name not in models:
                continue
            difficulty = self._question_difficulties.loc[model_name]
            difficulty.name = 'difficulty'
            group_df = group_df.merge(difficulty.reset_index(), on='id', how='left')

            try:
                coefs_df = self.fit_df(group_df)
            except GLMMFitError as err:
                logger.warning(f"{model_name}, {metric}: {err}")
                res = {'estimate': np.nan, 'p_value': 1, 'std_err': np.nan}
            else:
                res = dict(
                    estimate=coefs_df.loc[self._label, 'Estimate'],
                    p_value=coefs_df.loc[self._label, 'Pr(>|z|)'],
                    std_err=coefs_df.loc[self._label, 'Std. Error'],
                )


            glmm_results.append({
                'model': model_name,
                **res,
            })

        glmm_results_df = pd.DataFrame(glmm_results)

        if models is not None:
            models_with_results = glmm_results_df.model.unique()
            for requested_model_name in models:
                if requested_model_name not in models_with_results:
                    logger.warning(f"No data for model {requested_model_name}")

        glmm_results_df = glmm_results_df.set_index('model')
        return glmm_results_df
