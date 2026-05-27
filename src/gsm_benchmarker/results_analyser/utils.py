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


def pandas_to_latex(tab: pd.DataFrame, position: str = 't', clean_header: bool = True, **kwargs) -> str:
    """Return a LaTeX representation of a pandas DataFrame.

    Parameters
    ----------
    tab:
        DataFrame to convert.
    position:
        LaTeX float position argument (default: 't')
    clean_header:
        If True, sanitize column and index labels by escaping underscores.
    Additional keyword arguments are forwarded to pandas styling .to_latex.
    """
    tab = tab.copy()

    if clean_header:
        # escape LaTeX special characters
        tab.columns = [str(col).replace('_', '\\_') for col in tab.columns]
        tab.index = [str(idx).replace('_', '\\_') for idx in tab.index]

    tab.columns.name = tab.index.name
    tab.index.name = None

    # Generate the LaTeX string
    latex_code = tab.style.format(escape=None).to_latex(
        column_format='l' + 'c' * len(tab.columns), # No vertical bars
        position=position,          # ACL prefers 't' (top of page) or 'ht'
        hrules=True,           # triggers the booktabs lines (\toprule, etc.)
        **kwargs
    )

    return latex_code


def correct_p_values(p_values):
    """Apply Holm multiple-testing correction to a sequence of p-values.

    Returns the corrected p-values as a numpy array or pandas Series (matching
    the input type).
    """
    _, p_corrected, _, _ = multipletests(p_values, is_sorted=False, returnsorted=False, method='holm')
    if isinstance(p_values, pd.Series):
        p_corrected = pd.Series(p_corrected, index=p_values.index)
    return p_corrected
