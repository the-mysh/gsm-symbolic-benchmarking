import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_bars_and_p_bars(df: pd.DataFrame, value_col: str, p_value_col: str,
                          alpha: float = 0.05, projected_alpha: float | None = None, title: str | None = None,
                          colours: list[str] | None = None, models: list[str] | None = None):

    if colours is None:
        colours = ['lightblue', 'navy']

    if models is not None:
        df = df[np.isin(df.index.get_level_values('model'), models)]

    def prep_data(col):
        d = df[col].unstack(level='metric')
        d = d[['Standard accuracy', 'Discounted accuracy']]
        return d

    df_gap_closure = prep_data(value_col)
    df_p_values = prep_data(p_value_col)

    fig, axes = plt.subplots(2, 1, sharex='all', figsize=(12, 8))

    df_gap_closure.plot(ax=axes[0], kind='bar', color=colours)
    axes[0].set_ylabel(value_col.replace('_', ' ').capitalize())
    axes[0].axhline(0, color='k', lw=0.5)

    df_p_values.plot(ax=axes[1], kind='bar', color=colours)
    axes[1].set_xticklabels(df_p_values.index, rotation=45, ha='right')
    axes[1].axhline(alpha, ls='--', color='k', lw=0.5, label='alpha = 0.05')

    if projected_alpha is not None:
        axes[1].axhline(projected_alpha, ls=':', color='maroon', lw=0.5, label='equivalent alpha for full set')

    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('P value')

    for ax in axes:
        ax.legend(loc='upper right', frameon=True)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=7)

    if title:
        fig.suptitle(title)

    fig.tight_layout()

    return fig


def plot_stats(cs: pd.DataFrame, n_models: int = 20, titles: dict | None = None, title: str | None = None):
    colors =['limegreen', 'indianred', 'lightsteelblue']

    cs['not significant'] = n_models - cs.significant
    cs = cs.drop('significant', axis=1)
    cs = cs.rename(columns={'success': 'Improvement', 'failure': 'Deterioration', 'not significant': 'Change not significant'})

    n_plots = len(cs)
    fig, axes = plt.subplots(1, n_plots + 1, figsize=(12, 4))
    for param, ax in zip(cs.index, axes):
        wedges, _, _ = ax.pie(cs.loc[param], colors=colors, autopct=lambda p: str(round(p*n_models/100)))
        ax.set_title(titles.get(param, param))

    axes[-1].axis('off')
    axes[-1].legend(wedges, cs.columns, loc='center', title="No. if models which showed:")

    if title:
        fig.suptitle(title)

    return fig, cs
