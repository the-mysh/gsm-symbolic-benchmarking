"""Small utilities used for exporting tables and adjusting p-values."""

import pandas as pd
from statsmodels.stats.multitest import multipletests


# Labels used by higher-level analysers when running metrics for both
# standard and discounted correctness definitions.
METRIC_LABELS = {'correct': 'standard', 'correct_strict': 'discounted'}


def do_for_metrics(func):
    """Decorator to run an analysis function for multiple predefined metrics.

    If the wrapped function is called with metric=None, it will be executed
    for all metrics defined in METRIC_LABELS and the results concatenated into
    a single DataFrame with a top-level index named 'metric'. If metric is
    provided, the original function is called unchanged.
    """

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


def pandas_to_latex(tab: pd.DataFrame, position: str = 't', clean_header: bool = True, index=True,
                    column_format: str | None = None, **kwargs) -> str:
    """Return a LaTeX representation of a pandas DataFrame.

    Parameters
    ----------
    tab:
        DataFrame to convert.
    position:
        LaTeX float position argument (default: 't')
    clean_header:
        If True, sanitise column and index labels by escaping underscores.
    column_format:
        LaTeX column format string.
    Additional keyword arguments are forwarded to pandas styling .to_latex.
    """
    tab = tab.copy()

    if clean_header:
        # escape LaTeX special characters
        tab.columns = [str(col).replace('_', '\\_') for col in tab.columns]
        tab.index = [str(idx).replace('_', '\\_') for idx in tab.index]

    tab.columns.name = tab.index.name
    tab.index.name = None

    n_columns = len(tab.columns)
    if not index:
        n_columns -= 1

    # Generate the LaTeX string
    latex_code = tab.to_latex(
        column_format=column_format or ('l' + 'c' * n_columns),
        position=position,
        index=index,
        **kwargs
    )

    return latex_code


def format_float(precision=3, use_delta: bool = False, use_abs: bool = False):
    th = 10 ** (-precision)

    def wrapper(v):
        if not precision:
            return str(round(v))
        if abs(v) < th:
            pref = r"$|\cdot|$ " if use_abs else ""
            if use_delta:
                return pref + r"< $\delta$"
            return pref + f"< {th:.{precision}f}"
        return f"{v:.{precision}f}"

    return wrapper


def format_p_value(precision=3, alpha=0.05, projected_alpha: float | None = None, use_delta: bool = False):
    str_fmt = format_float(precision=precision, use_delta=use_delta)
    a = projected_alpha if projected_alpha is not None else alpha

    def wrapper(v):
        v_formatted = str_fmt(v)
        if v < a:
            return r"\textbf{" + v_formatted + "}"
        return v_formatted

    return wrapper


def correct_p_values(p_values):
    """Apply Holm multiple-testing correction to a sequence of p-values.

    Returns the corrected p-values as a numpy array or pandas Series (matching
    the input type).
    """
    _, p_corrected, _, _ = multipletests(p_values, is_sorted=False, returnsorted=False, method='holm')
    if isinstance(p_values, pd.Series):
        p_corrected = pd.Series(p_corrected, index=p_values.index)
    return p_corrected
