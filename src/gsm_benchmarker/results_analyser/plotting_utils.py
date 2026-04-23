import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb, rgb_to_hsv, hsv_to_rgb, rgb2hex
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from pathlib import Path
from typing import NamedTuple
import logging


logger = logging.getLogger(__name__)


def save_plot(label):
    def decorator(func):
        def wrapper(*args, save_prefix: str | Path | None = None, save_ext: str = "png", **kwargs):
            ret = func(*args, **kwargs)

            # Normalize only for figure collection; preserve wrapped return type.
            ret_items = ret if isinstance(ret, tuple) else (ret,)
            figures = [r for r in ret_items if isinstance(r, Figure)]

            idx_format = "_{}" if len(figures) > 1 else ""
            sep = "_"
            if isinstance(save_prefix, str) and save_prefix.endswith("/"):
                sep = ""

            for i, fig in enumerate(figures):
                if save_prefix is not None:
                    save_name = Path(f"{save_prefix}{sep}{label}{idx_format.format(i)}.{save_ext}").resolve()
                    fig.savefig(save_name)
                    logger.debug(f"Figure saved as: {save_name}")

            return ret

        return wrapper

    return decorator


class Colour:
    def __init__(self, c: str):
        self._value = to_rgb(c)

    @property
    def value(self):
        return rgb2hex(self._value)

    def lighten(self, factor: float = 0.5):
        h, s, v = rgb_to_hsv(self._value)

        v = v + factor * (1 - v)
        s = factor * s

        return rgb2hex(hsv_to_rgb([h, s, v]).tolist())


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


def _get_fig_size(n_models):
    return 10, max(n_models/5 + 2, 3)


@save_plot("bars")
def plot_bars_and_p_bars(df: pd.DataFrame, metric: str, value_col: str, p_value_col: str,
                         alpha: float = 0.05, projected_alpha: float | None = None, title: str | None = None,
                         bar_colour: str | None = None, models: list[str] | None = None,
                         model_order: list[str] | None = None, ylabel0: str | None = None):

    bar_colour = bar_colour or 'teal'

    if metric is not None:
        df = df.xs(metric, level='metric')

    if models is not None:
        df = df[np.isin(df.index.get_level_values('model'), models)]

    def prep_data(col):
        d = df[col]

        if model_order is not None:
            d = _sort_by_model(d.reset_index(), model_order).set_index('model')[col]

        return d

    data_val = prep_data(value_col)
    df_p_values = prep_data(p_value_col)

    fig, axes = plt.subplots(1, 2, sharey='all', figsize=_get_fig_size(len(df)))

    data_val.plot(ax=axes[0], kind='barh', color=bar_colour, legend=False)
    axes[0].set_xlabel(ylabel0 if ylabel0 is not None else value_col.replace('_', ' ').capitalize())
    axes[0].axvline(0, color='k', lw=0.5)

    df_p_values.plot(ax=axes[1], kind='barh', color=bar_colour, legend=False)

    if df_p_values.max() >= 0.001 * alpha:
        handles = [axes[1].axvline(alpha, ls='--', color='navy', lw=1, label=f'alpha = {alpha:.2f}')]

        if projected_alpha is not None:
            l = axes[1].axvline(projected_alpha, ls=':', color='royalblue', lw=1,
                                label=f'projected alpha = {projected_alpha:.2f}')
            handles.append(l)
            axes[1].legend(frameon=True, handles=handles, fontsize=8)

    axes[0].set_ylabel('Model')
    axes[1].set_xlabel('P value')

    for ax in axes:
        ax.axvline(0, color='k', lw=1, zorder=1)
        for container in ax.containers:
            labels = [
                f"{v:.3f}" if v >= 0.001 or v == 0 else f"{v:.1e}"
                for v in container.datavalues
            ]

            ax.bar_label(container, labels=labels, fontsize=7)
        ax.margins(x=0.1)  # make sure label of lowest bar does not overlap with y-axis labels - give more margin

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


@save_plot("difficulty_matrix")
def plot_question_success_rate_matrix(df, title: str | None = None):
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

    if title is not None:
        fig.suptitle(title, fontsize=16, y=0.95)

    return fig


@save_plot("question_difficulty_histogram")
def plot_question_difficulty_histogram(difficulties, n_levels: int = 10, color: str | None = None):
    g = sns.displot(data=difficulties, kde=False,
                    binwidth=1 / n_levels, binrange=(0, 1),
                    edgecolor='white', color=color or 'tab:blue',
                    aspect=1.5
                    )

    return g.figure


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


def _prepare_odds_ratios_data(df: pd.DataFrame, metric: str | None = None, projected_alpha: float | None = None,
                              model_order: list[str] | None = None, sort_models: bool = False
                              ) -> tuple[pd.DataFrame, dict[str, SignificancePoint], list[str] | None]:

    p_thresholds = _define_significance_points(projected_alpha)

    if metric is not None:
        df = df.xs(metric, level='metric')
    df_plot = df.copy()

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


@save_plot("odds_ratios")
def plot_models_odds_ratios(df, metric: str | None = None, projected_alpha: float | None = None,
                            model_order: list[str] | None = None, log_scale: bool = False, sort_models: bool = False,
                            title: str | None = None):

    df_plot, p_thresholds, model_order = _prepare_odds_ratios_data(
        df, metric=metric, projected_alpha=projected_alpha, model_order=model_order, sort_models=sort_models)

    fig, ax = plt.subplots(figsize=_get_fig_size(len(df_plot)))
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


def plot_for_metrics(func):
    def wrapper(df: pd.DataFrame, *args, **kwargs):
        save_prefix_in_kwargs = 'save_prefix' in kwargs and kwargs['save_prefix'] is not None
        save_prefix = kwargs.pop('save_prefix', None)


        if 'metric' in df.index.names:
            ret = []
            for metric in df.index.get_level_values('metric').unique()[::-1]:
                sp = {'save_prefix': f"{save_prefix}_{metric}"} if save_prefix_in_kwargs else {}
                r = func(df, *args, metric=metric, **kwargs, **sp)
                if r is not None:
                    ret.extend(r) if isinstance(r, tuple) else ret.append(r)
        else:
            sp = {'save_prefix': save_prefix} if save_prefix_in_kwargs else {}
            ret = func(df, *args, metric=None, **kwargs, **sp)

        return ret

    return wrapper


@plot_for_metrics
def plot_glmm(df: pd.DataFrame, bars_value_col: str, bars_value_ylabel: str | None = None, metric: str | None = None,
              bar_colour: str | None = None, title: str | None = None, save_prefix: str | Path | None = None, **kwargs):

    metric_text = f"\n{metric} accuracy" if metric else ""

    f1, model_order = plot_models_odds_ratios(
        df, metric, log_scale=True, sort_models=True, save_prefix=save_prefix, **kwargs,
        title=f"{title} - odds ratios{metric_text}" if title else None,
    )

    f2 = plot_bars_and_p_bars(
        df, metric, value_col=bars_value_col, p_value_col='p_value', bar_colour=bar_colour,
        model_order=model_order, ylabel0=bars_value_ylabel, save_prefix=save_prefix, **kwargs,
        title=f"{title} - magnitude and significance{metric_text}" if title else None,
    )

    return f1, f2


@plot_for_metrics
@save_plot("acc_change_distribution")
def plot_acc_change_distribution(df: pd.DataFrame, col_name: str = 'acc_change', metric: str | None = None,
                                 models: list[str] | None = None, color: str | None = None):
    if metric is not None:
        df = df.xs(metric, level='metric')

    if models is not None:
        df = df.loc[models]
    df = df.reset_index()

    new_col_name = col_name.replace('_', ' ').capitalize()
    df.rename(columns={col_name: new_col_name}, inplace=True)

    width = 0.1

    # expand the range slightly and offset it by half the width to ensure 0 is centered
    data_min = df[new_col_name].min()
    data_max = df[new_col_name].max()
    start = (np.floor(data_min / width) * width) - (width / 2)
    end = (np.ceil(data_max / width) * width) + (width / 2)

    # This creates a grid of histograms automatically
    g = sns.displot(data=df, x=new_col_name, col='model', col_wrap=2, kde=True,
                    binwidth=width, binrange=(start, end),
                    edgecolor='white', color=color or 'rebeccapurple',
                    facet_kws={'sharex': True, 'sharey': True},
                    height=3, aspect=1.5)
    g.refline(x=0, color='k', linestyle='--', lw=1)

    return g.figure


@save_plot("prompts")
def plot_prompt_format_comparison(plot_df: pd.DataFrame, selected_models: list[str], prompt_order: list[str],
                                  prompt_labels: dict[str, str], prompt_colours: dict[str, str],
                                  title: str | None = None, mean_ylabel: str = 'Mean accuracy',
                                  variant_ylabel: str = 'Symbolic performance delta',
                                  significance_threshold: float = 0.05):
    required_cols = {'model', 'prompt', 'mean_accuracy', 'accuracy_change', 'p_value'}
    missing = required_cols - set(plot_df.columns)
    if missing:
        raise ValueError(f"plot_df is missing required columns: {', '.join(sorted(missing))}")

    plot_df = plot_df.copy()

    acc_pivot = (
        plot_df
        .pivot(index='model', columns='prompt', values='mean_accuracy')
        .reindex(selected_models)
        .reindex(columns=prompt_order)
    )
    var_pivot = (
        plot_df
        .pivot(index='model', columns='prompt', values='accuracy_change')
        .reindex(selected_models)
        .reindex(columns=prompt_order)
    )
    sig_pivot = (
        plot_df
        .pivot(index='model', columns='prompt', values='p_value')
        .reindex(selected_models)
        .reindex(columns=prompt_order)
    )

    x = np.arange(len(selected_models))
    bar_width = 0.18
    offset_center = (len(prompt_order) - 1) / 2

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    for i, prompt_key in enumerate(prompt_order):
        offsets = x + (i - offset_center) * bar_width
        color = prompt_colours[prompt_key]

        bars_acc = ax1.bar(
            offsets,
            acc_pivot[prompt_key].to_numpy(),
            width=bar_width,
            label=prompt_labels[prompt_key],
            color=color,
        )
        ax1.bar_label(bars_acc, fmt='%.3f', fontsize=8, padding=2)

        bars_var = ax2.bar(
            offsets,
            var_pivot[prompt_key].to_numpy(),
            width=bar_width,
            color=color,
        )
        ax2.bar_label(bars_var, fmt='%.3f', fontsize=8, padding=2)

        pvals = sig_pivot[prompt_key].to_numpy()
        for bar, p_value in zip(bars_var, pvals):
            if np.isfinite(p_value) and p_value > significance_threshold:
                bar.set_hatch('///')
                bar.set_edgecolor('black')
                bar.set_linewidth(0.8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(selected_models, rotation=15, ha='right')
    ax1.set_ylim(0, 1)
    ax1.set_ylabel(mean_ylabel)

    ax2.set_xticks(x)
    ax2.set_xticklabels(selected_models, rotation=15, ha='right')
    ax2.set_ylabel(variant_ylabel)
    ax2.set_xlabel('Model')

    for ax in (ax1, ax2):
        ax.axhline(0, color='black', linewidth=0.5)


    prompt_handles, prompt_labels_for_legend = ax1.get_legend_handles_labels()
    hatch_label = f'Not significant (p > {significance_threshold:.2f})'
    hatch_patch = Patch(facecolor='white', edgecolor='black', hatch='///', label=hatch_label)
    ax2.legend(
        handles=prompt_handles + [hatch_patch],
        labels=prompt_labels_for_legend + [hatch_label],
        title='Prompt / significance',
        frameon=True,
        fontsize=8,
    )

    if title:
        fig.suptitle(title, y=1.02)

    fig.tight_layout()

    return fig

