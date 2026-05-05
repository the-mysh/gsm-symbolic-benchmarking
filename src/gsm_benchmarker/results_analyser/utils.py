import pandas as pd


def pandas_to_latex(tab: pd.DataFrame, position: str = 't', clean_header: bool = True, **kwargs) -> str:
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
