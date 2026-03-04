import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from typing import NamedTuple


class SignificancePoint(NamedTuple):
    threshold: float | None
    is_drop: bool | None
    colour: str
    label: str


def _sort_by_model(df, model_order: list[str]):
    return df.sort_values(
        by='model',
        key=lambda col: col.map({model: index for index, model in enumerate(model_order)})
    ).reset_index(drop=True)


def plot_bars_and_p_bars(df: pd.DataFrame, metric: str, value_col: str, p_value_col: str,
                         alpha: float = 0.05, projected_alpha: float | None = None, title: str | None = None,
                         colour: str | None = None, models: list[str] | None = None,
                         model_order: list[str] | None = None, ylabel0: str | None = None):

    colour = colour or 'navy'

    df = df.xs(metric, level='metric')

    if models is not None:
        df = df[np.isin(df.index.get_level_values('model'), models)]

    def prep_data(col):
        d = df[col]

        if model_order is not None:
            d = _sort_by_model(d.reset_index(), model_order[::-1]).set_index('model')

        return d

    data_val = prep_data(value_col)
    df_p_values = prep_data(p_value_col)

    fig, axes = plt.subplots(2, 1, sharex='all', figsize=(12, 8))

    data_val.plot(ax=axes[0], kind='bar', color=colour)
    axes[0].set_ylabel(ylabel0 if ylabel0 is not None else value_col.replace('_', ' ').capitalize())
    axes[0].axhline(0, color='k', lw=0.5)

    df_p_values.plot(ax=axes[1], kind='bar', color=colour)
    axes[1].set_xticklabels(df_p_values.index, rotation=45, ha='right')
    axes[1].axhline(alpha, ls='--', color='k', lw=0.5, label=f'alpha = {alpha:.2f}')

    if projected_alpha is not None:
        axes[1].axhline(projected_alpha, ls=':', color='maroon', lw=0.5, label=f'projected alpha = {projected_alpha:.2f}')

    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('P value')
    axes[1].legend(loc='upper right', frameon=True)

    for ax in axes:
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


def _define_significance_points(projected_alpha: float | None):
    pp = " (p < {})"
    p_thresholds = {
        'strong_drop': SignificancePoint(0.01, True, 'brown', f'Strong drop' + pp),
        'significant_drop': SignificancePoint(0.05, True, 'sandybrown', 'Significant drop' + pp),
        'potentially_significant_drop': SignificancePoint(
            projected_alpha, True, 'khaki', 'Potentially significant drop' + pp),
        'not_significant': SignificancePoint(1, None, 'darkgray', f'Not significant'),
        'potentially_significant_rise': SignificancePoint(
            projected_alpha, False, 'palegreen', 'Potentially significant rise' + pp),
        'significant_rise': SignificancePoint(0.05, False, 'limegreen','Significant rise' + pp),
        'strong_rise': SignificancePoint(0.01, False, 'darkgreen', 'Strong rise' + pp)
    }

    return p_thresholds


def _prepare_odds_ratios_data(df: pd.DataFrame, metric: str, projected_alpha: float | None = None,
                              model_order: list[str] | None = None, sort_models: bool = False
                              ) -> tuple[pd.DataFrame, dict[str, SignificancePoint], list[str] | None]:

    p_thresholds = _define_significance_points(projected_alpha)

    df_plot = df.xs(metric, level='metric').copy()

    def get_colour(row):
        default_colour = p_thresholds["not_significant"].colour
        p = row['p_value']
        is_drop = bool(row['estimate'] < 0)

        if np.isnan(p):
            return default_colour
        for name, point in p_thresholds.items():
            if point.threshold is None or point.is_drop is None:
                continue
            if p < point.threshold and is_drop is point.is_drop:
                return point.colour
        return default_colour

    df_plot['colour'] = df_plot.apply(get_colour, axis=1)

    # compute odds ratios and 95% confidence intervals
    estimate = df_plot.estimate
    err = df_plot['std_err']
    err_threshold = estimate.abs().max() * 0.5
    df_plot['odds_ratio'] = np.exp(estimate)
    df_plot['odds_ratio_plot'] = np.where(np.isnan(df_plot.odds_ratio), 1, df_plot.odds_ratio)

    std_err_clipped = np.minimum(err, err_threshold)  # clip for plotting
    f = 1.96
    ci_lower_log = estimate - f * std_err_clipped
    ci_upper_log = estimate + f * std_err_clipped
    df_plot['ci_lower_or'] = np.exp(ci_lower_log)
    df_plot['ci_upper_or'] = np.exp(ci_upper_log)
    df_plot['ci_clipped'] = np.where(err > err_threshold, True, False)

    if model_order is None:
        if sort_models:
            model_order = df_plot.sort_values(by='odds_ratio_plot', ascending=False).index.to_list()

    df_plot = df_plot.reset_index()

    if model_order is not None:  # after sort_models check
        df_plot = _sort_by_model(df_plot, model_order)

    return df_plot, p_thresholds, model_order


def plot_models_odds_ratios(df, metric, projected_alpha: float | None = None, model_order: list[str] | None = None,
                            log_scale: bool = False, sort_models: bool = False, title: str | None = None):

    df_plot, p_thresholds, model_order = _prepare_odds_ratios_data(
        df, metric=metric, projected_alpha=projected_alpha, model_order=model_order, sort_models=sort_models)

    fig, ax = plt.subplots(figsize=(10, max(len(df_plot)/5 + 2, 3)))
    ci_colour = 'darkgrey'

    # plot CIs and coloured dots
    for i, row in df_plot.iterrows():
        y = row['model']

        if np.isnan(row['odds_ratio']):
            ax.scatter(x=[1], y=[y], marker='x', color='k', lw=0.5, s=50)
            continue

        # draw CIs
        ax.hlines(y, xmin=row['ci_lower_or'], xmax=row['ci_upper_or'], color=ci_colour, lw=2)
        if row["ci_clipped"]:
            # mark that the errors are in fact bigger - clipped here
            ax.plot(row['ci_lower_or'], y, '<', c=ci_colour)
            ax.plot(row['ci_upper_or'], y, '>', c=ci_colour)
        else:
            ax.plot([row['ci_lower_or'], row['ci_upper_or']], [y, y], '|', c=ci_colour)

        # draw the dot using the dynamically assigned colour
        ax.scatter(x=row['odds_ratio'], y=y, color=row['colour'], s=80, zorder=2, ec='black', lw=0.5)

    ax.axvline(x=1, color='black', linestyle='--', linewidth=1.2, zorder=0)  # line of no effect

    if log_scale:
        ax.set_xscale('log')

    ax.set_xlabel('Odds ratio' +  (' (log scale)' if log_scale else ''))

    # legend
    point_colours = df_plot.colour.unique()
    legend_elements = [
        Line2D(
            [0], [0], marker='o', c='darkgrey', mec='black', mew=0.5, ms=10,
            mfc=point.colour, label=point.label.format(point.threshold)
        ) for point in p_thresholds.values() if point.colour in point_colours
    ]
    if np.isnan(df_plot.odds_ratio).any():
        legend_elements.append(
            Line2D([0], [0], marker='x', c='k', mec='black', mew=0.5, ms=8, lw=0, label="Could not compute"))

    ax.legend(handles=legend_elements, title="Significance", frameon=True, fontsize=8)

    ax.set_ylabel('Model')

    if title:
        fig.suptitle(title)

    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    return fig, model_order


def plot_glmm(df: pd.DataFrame, bars_value_col: str, bars_value_ylabel: str | None = None,
              bar_colour: str | None = None, title: str | None = None, **kwargs):

    figs = []

    for metric in df.index.get_level_values('metric').unique():
        print(f"{metric.capitalize()} accuracy")

        fig_or, model_order = plot_models_odds_ratios(
            df, metric, log_scale=True, sort_models=True, **kwargs,
            title=f"{title} - odds ratios\n{metric} accuracy" if title else None
        )

        fig_bars = plot_bars_and_p_bars(
            df, metric, value_col=bars_value_col, p_value_col='p_value', colour=bar_colour,
            model_order=model_order, ylabel0=bars_value_ylabel, **kwargs,
            title=f"{title} - magnitude and significance\n{metric} accuracy" if title else None
        )

        figs.extend((fig_or, fig_bars))

    return figs
