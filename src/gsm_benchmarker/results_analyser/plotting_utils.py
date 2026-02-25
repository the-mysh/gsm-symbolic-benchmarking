import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D



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


def plot_question_success_rate_matrix(df):
    n_models, n_questions = df.shape

    # Calculate marginals & sort
    difficulty_per_model_left_out = df.sum(axis=1) / n_questions
    difficulty_per_question = df.sum(axis=0) / n_models
    sorted_models = difficulty_per_model_left_out.sort_values(ascending=False).index  # worst to best
    sorted_questions = difficulty_per_question.sort_values(ascending=False).index  # hardest to easiest
    df_sorted = df.loc[sorted_models, sorted_questions]  # reorder dataframe based on the sorting

    with plt.style.context('seaborn-v0_8-whitegrid'):
        fig, (ax_top, ax_heatmap) = plt.subplots(
            2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 3]}, sharex='all')

        # heatmap
        cmap = sns.color_palette("mako_r", as_cmap=True)
        sns.heatmap(df_sorted, cmap=cmap, cbar=False, ax=ax_heatmap,
                    xticklabels=False, yticklabels=True)

        ax_heatmap.set_xlabel("Questions (sorted hardest to easiest)", fontsize=12, labelpad=10)
        ax_heatmap.set_ylabel("Models left out", fontsize=12, labelpad=10)

        norm = plt.Normalize(0, 1)

        # top - bar chart for question success rate
        ax_top.bar(
            np.arange(n_questions) + 0.5,
            difficulty_per_question.loc[sorted_questions] * 100,
            color=cmap(norm(difficulty_per_question.loc[sorted_questions])),  # matching success rate
            width=1.0
        )

        ax_top.set_ylabel("Question difficulty\n(% models failed)", fontsize=10)
        ax_top.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        for spine in ['top', 'right', 'bottom']:
            ax_top.spines[spine].set_visible(False)
        ax_top.grid(False)
        ax_top.set_ylim(0, 100)

    fig.suptitle("Leave-one-(model-)out question difficulty", fontsize=16, y=0.95)

    return fig


def plot_models_odds_ratios(df, projected_alpha: float | None = None, sort_models: bool = True):
    p_thresholds = {
        'strong': (0.01, 'brown', 'Strong drop (p < {})'),
        'significant': (0.05, 'orange', 'Significant drop (p < {})'),
        'potentially_significant': (projected_alpha, 'gold', 'Potentially significant drop (p < {})'),
        'not_significant': (1, 'darkgray', f'Not significant')
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    if sort_models:
        # Sort the dataframe so the best performing models (highest OR) are at the top
        df_plot = df.sort_values('odds_ratio', ascending=True)
    else:
        df_plot = df.copy()

    def get_color(p):
        for name, (th, c, _) in p_thresholds.items():
            if th is None:
                continue
            if p < th:
                return c
        return p_thresholds["not_significant"][0]

    df_plot['color'] = df_plot['p_value'].apply(get_color)

    # plot CIs and coloured dots
    for i, row in df_plot.iterrows():
        # draw CIs
        ax.hlines(y=row['model'], xmin=row['ci_lower_or'], xmax=row['ci_upper_or'],
                  color='darkgrey', linewidth=2, zorder=1)

        # draw the dot using the dynamically assigned colour
        ax.scatter(x=row['odds_ratio'], y=row['model'],
                   color=row['color'], s=100, zorder=2, edgecolor='black', linewidth=0.5)

    ax.axvline(x=1, color='black', linestyle='--', linewidth=1.2, zorder=0)  # line of no effect

    ax.set_xlabel('Odds ratio')
    ax.set_ylabel('Model')
    ax.set_title('Effect of question templates on accuracy', pad=15)

    # legend
    legend_elements = [
        Line2D(
            [0], [0], marker='o', c='darkgrey', mec='black', mew=0.5, mfc=c, ms=10, label=desc.format(th)
        ) for th, c, desc in p_thresholds.values() if th is not None
    ]
    ax.legend(handles=legend_elements, title="Significance", frameon=True, fontsize=8)

    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    return fig
