"""Plotting utilities used by the results analyser.

This module provides a set of reusable plotting functions and small helper
classes used for consistent visualisations across the benchmarking analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb, rgb_to_hsv, hsv_to_rgb, rgb2hex
from matplotlib.figure import Figure
from matplotlib.patches import Patch
import matplotlib.ticker as mtick
from matplotlib import rc_context
from pathlib import Path
from typing import NamedTuple
import logging


logger = logging.getLogger(__name__)


VARIANT_COLOURS = {
    'GSM-Base': 'mediumslateblue',
    'GSM-Variants': 'darksalmon'
}


def save_plot(*labels):
    """Decorator to optionally save figures returned by a plotting function.

    The decorated function can return one or more matplotlib Figure objects
    (or other values). Figures are saved to disk using the provided labels
    when `save_prefix` is passed to the call.
    """

    def decorator(func):
        def wrapper(*args, save_prefix: str | Path | None = None, save_ext: str = "png", **kwargs):
            ret = func(*args, **kwargs)

            # Normalize only for figure collection; preserve wrapped return type.
            ret_items = ret if isinstance(ret, tuple) else (ret,)
            figures = [r for r in ret_items if isinstance(r, Figure)]

            if (nl := len(labels)) < (nf := len(figures)):
                raise ValueError(f"Got {nf} figures, but only {nl} labels")

            sep = "_"
            if isinstance(save_prefix, str) and save_prefix.endswith("/"):
                sep = ""

            for label, fig in zip(labels, figures):
                if save_prefix is not None:
                    save_name = Path(f"{save_prefix}{sep}{label}.{save_ext}").resolve()
                    fig.savefig(save_name)
                    logger.debug(f"Figure saved as: {save_name}")

            return ret

        return wrapper

    return decorator


class Colour:
    """Small colour helper that stores a matplotlib colour and emits hex strings.

    Provides convenience methods to lighten or darken a colour for aesthetic
    variations in plots.
    """

    def __init__(self, c: str):
        self._value = to_rgb(c)

    @property
    def value(self):
        return rgb2hex(self._value)

    @staticmethod
    def _increase(value, factor):
        return min(value + factor * (1 - value), 1)

    @staticmethod
    def _decrease(value, factor):
        return max(value - factor * (1 - value), 0)

    def lighten(self, factor: float = 0.5) -> "Colour":
        h, s, v = rgb_to_hsv(self._value)

        v = self._increase(v, factor)
        s = self._decrease(s, factor)

        return Colour(rgb2hex(hsv_to_rgb([h, s, v]).tolist()))

    def darken(self, factor: float = 0.5) -> "Colour":
        h, s, v = rgb_to_hsv(self._value)

        v = self._decrease(v, factor)
        s = self._increase(v, factor)

        return Colour(rgb2hex(hsv_to_rgb([h, s, v]).tolist()))


class SignificancePoint(NamedTuple):
    """Descriptor for plotting significance legend entries.

    Fields are:
    - threshold: p-value threshold
    - is_drop: whether the effect corresponds to a drop (True) or rise (False)
    - colour: plotting colour
    - label: text label template
    """

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
    return 10, max(n_models/5 + 1.5, 2.5)


@save_plot("bars")
def plot_bars_and_p_bars(df: pd.DataFrame, metric: str, value_col: str, p_value_col: str,
                         alpha: float = 0.05, projected_alpha: float | None = None, title: str | None = None,
                         bar_colour: str | None = None, models: list[str] | None = None,
                         model_order: list[str] | None = None, value_label: str | None = None):
    """Create side-by-side bar plots showing effect magnitude and p-values.

    Returns a Figure containing two horizontally aligned subplots: the left
    shows the magnitude (value_col) and the right the p-values for each model.
    """

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
    axes[0].set_xlabel(value_label if value_label is not None else value_col.replace('_', ' ').capitalize())
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
        ax.margins(x=0.1)  # make sure label of lowest bar does not overlap with y-axis labels - give more margin

    add_bar_labels(axes[0], precision=1, fontsize=7)
    add_bar_labels(axes[1], precision=3, fontsize=7)

    if title:
        fig.suptitle(title)

    fig.tight_layout()

    return fig


def add_bar_labels(ax, precision: int = 3, fontsize: int = 7):
    """Add numeric labels to bars in a matplotlib Axes.

    The precision parameter controls formatting; precision==0 yields integer
    labels, positive values produce fixed-point formatting and very small
    numbers use scientific notation.
    """
    if precision == 0:
        fmt = lambda v: f"{v:d}"
    elif precision > 0:
        def fmt(v):
            if not v:
                return "0.0"
            if abs(v) >= 10**(-precision):
                return f"{v:.{precision}f}"
            return f"{v:.1e}"
    else:
        raise ValueError("Precision cannot be negative")

    for container in ax.containers:
        labels = [fmt(val) for val in container.datavalues]
        ax.bar_label(container, labels=labels, fontsize=fontsize)


def plot_stats(cs: pd.DataFrame, n_models: int = 20, titles: dict | None = None, title: str | None = None):
    """Plot summary pie charts for improvement / deterioration statistics.

    Returns a tuple (figure, transformed_dataframe) where the dataframe has the
    counts used to create the plot.
    """
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


def _define_significance_points(projected_alpha: float | None):
    pp = " (p < {})"
    p_thresholds = {
        'strong_drop': SignificancePoint(0.01, True, 'brown', f'Strong drop' + pp),
        'significant_drop': SignificancePoint(0.05, True, 'sandybrown', 'Significant drop' + pp),
        'potentially_significant_drop': SignificancePoint(
            projected_alpha, True, 'khaki', 'Potentially significant drop' + pp),
        'strong_rise': SignificancePoint(0.01, False, 'darkgreen', 'Strong rise' + pp),
        'significant_rise': SignificancePoint(0.05, False, 'limegreen', 'Significant rise' + pp),
        'potentially_significant_rise': SignificancePoint(
            projected_alpha, False, 'palegreen', 'Potentially significant rise' + pp),
        'not_significant': SignificancePoint(1, None, 'darkgray', f'Not significant')
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
    df_plot['odds_ratio_plot'] = np.where(np.isnan(df_plot.odds_ratio), 1, df_plot.odds_ratio)

    f = 1.96
    ci_lower_log = estimate - f * err
    ci_upper_log = estimate + f * err
    df_plot['ci_lower_log_or'] = ci_lower_log
    df_plot['ci_upper_log_or'] = ci_upper_log

    if model_order is None:
        if sort_models:
            model_order = df_plot.sort_values(by='odds_ratio_plot', ascending=False).index.to_list()

    df_plot = df_plot.reset_index()

    if model_order is not None:  # after sort_models check
        df_plot = _sort_by_model(df_plot, model_order)

    return df_plot, p_thresholds, model_order


@save_plot("odds_ratios")
def plot_models_odds_ratios(df, metric: str | None = None, projected_alpha: float | None = None,
                            model_order: list[str] | None = None, sort_models: bool = False,
                            title: str | None = None):
    """Plot log odds ratios with 95% CI for each model.

    Returns a tuple (figure, model_order) where model_order is the effective
    order of models used to generate the plot.
    """

    df_plot, p_thresholds, model_order = _prepare_odds_ratios_data(
        df, metric=metric, projected_alpha=projected_alpha, model_order=model_order, sort_models=sort_models)

    fig, ax = plt.subplots(figsize=_get_fig_size(len(df_plot)))
    ci_colour = 'darkgrey'

    # plot CIs and coloured dots
    for i, row in df_plot.iterrows():
        y = row['model']

        if np.isnan(row['estimate']):
            ax.scatter(x=[1], y=[y], marker='x', color='k', lw=0.5, s=50)
            continue

        # draw CIs
        ax.hlines(y, xmin=row['ci_lower_log_or'], xmax=row['ci_upper_log_or'], color=ci_colour, lw=2)
        ax.plot([row['ci_lower_log_or'], row['ci_upper_log_or']], [y, y], '|', c=ci_colour, ms=15)

        # draw the dot using the dynamically assigned colour
        ax.scatter(x=row['estimate'], y=y, color=row['colour'], s=80, zorder=2, ec='black', lw=0.5)

    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.2, zorder=0)  # line of no effect

    ax.set_xlabel('Log Odds Ratio')

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
    """Decorator that runs plotting functions separately for each metric level.

    If the supplied DataFrame has a 'metric' level in its index, the wrapped
    function will be executed once per-metric and return a list of figures.
    Otherwise the function is called once with metric=None.
    """

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
              bar_colour: str | None = None, title: str | None = None, save_prefix: str | Path | None = None,
               model_order: list[str] | None = None, **kwargs):
    """Compose GLMM plots: odds ratios and bar/p-value panels.

    Returns a tuple of Figures (odds ratios figure, bar/p-value figure).
    """

    metric_text = f"\n{metric} accuracy" if metric else ""

    f1, model_order = plot_models_odds_ratios(
        df, metric, sort_models=True, save_prefix=save_prefix, model_order=model_order, **kwargs,
        title=f"{title} - odds ratios{metric_text}" if title else None,
    )

    f2 = plot_bars_and_p_bars(
        df, metric, value_col=bars_value_col, p_value_col='p_value', bar_colour=bar_colour,
        model_order=model_order, value_label=bars_value_ylabel, save_prefix=save_prefix, **kwargs,
        title=f"{title} - magnitude and significance{metric_text}" if title else None,
    )

    return f1, f2


@plot_for_metrics
@save_plot("acc_change_distribution")
def plot_acc_change_distribution(df: pd.DataFrame, col_name: str = 'acc_diff', label: str | None = None, metric: str | None = None,
                                 models: list[str] | None = None, color: str | None = None):
    """Plot per-model distributions of accuracy change as small multiples.

    Returns the matplotlib Figure created by seaborn's displot.
    """
    if metric is not None:
        df = df.xs(metric, level='metric')

    if models is not None:
        df = df.loc[models]
    df = df.reset_index()

    new_col_name = label or col_name.replace('_', ' ').capitalize()
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


@save_plot("prompts_variant_effect", "prompts_number_effect")
def plot_prompt_comparison(all_prompts_summary: pd.DataFrame, colours: dict[str, str], models: list[str] | None = None,
                           hatch_lw: int = 2, add_bar_labels: bool = False, x_labels_rotation: float = 0,
                           x_labels_ha='center'):
    """Plot two summary figures comparing prompt formats and effects.

    Returns two Figures: one showing accuracy and delta statistics, another
    showing log-odds and number-effect statistics.
    """
    if models:
        all_prompts_summary = all_prompts_summary[models]

    prompts = all_prompts_summary.index.get_level_values('prompt').unique().tolist()
    or_label = r"log OR"

    def prep_data(q):
        data = all_prompts_summary.xs(q, level='quantity')
        data = data.reindex(prompts, fill_value=None)
        data = data.transpose()
        return data

    def plot_quantity(quantity, ax, title, mask_quantity=None, precision: int = 3, ylabel: str | None = None, **kwargs):
        data = prep_data(quantity)
        mask = prep_data(mask_quantity) if mask_quantity else None

        with rc_context({'hatch.linewidth': hatch_lw}):
            data.plot.bar(ax=ax, legend=False, **kwargs, edgecolor='white')

            for i, container in enumerate(ax.containers):
                if add_bar_labels:
                    heights = [bar.get_height() for bar in container.patches]
                    labels = [f'{height:.{precision}f}' if not (height is None or np.isnan(height)) else '' for height
                              in heights]
                    ax.bar_label(container, labels=labels, fontsize=6, padding=1)

                if mask is not None:
                    for bar, sig in zip(container.patches, mask[mask.columns[i]]):
                        if not sig:
                            bar.set_hatch('///')

            ax.set_title(title)
            ax.axhline(0, c='k', lw=0.5)

            if ylabel is not None:
                ax.set_ylabel(ylabel, fontsize=8)

    # Reusable figure builder
    def create_figure(specs, figsize, bottom_rect, legend_title=None):
        fig, axes = plt.subplots(len(specs), 1, figsize=figsize, sharex='all')

        # Ensure axes is iterable even if there's only 1 subplot
        if len(specs) == 1:
            axes = [axes]

        # Iterate over the provided configurations to build each subplot
        for ax, spec in zip(axes, specs):
            plot_quantity(ax=ax, **spec)

        axes[-1].set_xticklabels(axes[-1].get_xticklabels(), rotation=x_labels_rotation, ha=x_labels_ha)
        axes[-1].set_xlabel("Model")

        # Create standard handles/labels and append the shared hatch patch
        handles, labels = axes[0].get_legend_handles_labels()
        with rc_context({'hatch.linewidth': hatch_lw}):
            hatch_patch = Patch(facecolor='grey', edgecolor='white', hatch='///', label=r"effect n.s. (p > .05)")
        handles.append(hatch_patch)
        labels.append(hatch_patch.get_label())

        fig.legend(handles, labels, title=legend_title, loc='lower center', ncol=6, frameon=True)

        # We adjust bottom_rect slightly depending on subplot count to leave room for the legend
        fig.tight_layout(rect=(0, bottom_rect, 1, 1))

        return fig

    # Configuration for the first plot (2 subplots)
    plot1_specs = [
        dict(quantity='GSM8K_acc', title='Mean accuracy on GSM-Base', color=colours, precision=1, ylabel="Accuracy, %"),
        dict(quantity='main_acc', title='Mean accuracy on GSM-Variants', color=colours, precision=1,
             ylabel="Accuracy, %"),
        dict(quantity='delta_symb_acc_diff', title=r'Variant performance delta', color=colours,
             mask_quantity='delta_symb_significant', precision=2, ylabel="Accuracy change, pp"),
    ]

    # Configuration for the second plot (3 subplots)
    plot2_specs = [
        dict(quantity='delta_symb_log_or', title=r'Variant effect - log odds ratio', color=colours,
             mask_quantity='delta_symb_significant', precision=2, ylabel=or_label),
        dict(quantity='number_effect_log_or', title=r'Number effect - log odds ratio', color=colours,
             mask_quantity='number_effect_significant', precision=2, ylabel=or_label),
        dict(quantity='delta_symb_ne_log_or', title=r'Number-effect-corrected variant effect - log odds ratio',
             color=colours,
             mask_quantity='delta_symb_ne_significant', precision=2, ylabel=or_label)
    ]

    # Generate both figures with adjusted heights and bottom margins
    fig1 = create_figure(plot1_specs, figsize=(10, 8), bottom_rect=0.06,
                         legend_title='Prompt formats & significance of variant effect')
    fig2 = create_figure(plot2_specs, figsize=(10, 8), bottom_rect=0.06,
                         legend_title='Prompt formats & significance of variant and number effects')

    return fig1, fig2


@save_plot("prompt_acc_evolution")
def plot_prompt_acc_evolution(all_prompts_summary, colours: dict[str, str], models: list[str] | None = None,
                              n_cols: int = 2, sharex='all', sharey='all', equal_aspect: bool = True, figsize=(10, 8), bottom_margin=.05):
    """Plot prompt-level accuracy evolution per model as small multiples.

    Returns a Figure with subplots arranged in a grid showing for each model
    how prompt formats compare (GSM-Base accuracy vs variant delta).
    """
    if models:
        all_prompts_summary = all_prompts_summary[models]
    else:
        models = all_prompts_summary.columns.values.tolist()

    n_models = len(models)
    n_rows = n_models // n_cols + n_models % n_cols

    x_data = all_prompts_summary.xs('GSM8K_acc', level='quantity')
    y_data = all_prompts_summary.xs('delta_symb_acc_diff', level='quantity')
    sig_data = all_prompts_summary.xs('delta_symb_significant', level='quantity')

    fig, axes = plt.subplots(n_rows, n_cols, sharex=sharex, sharey=sharey, figsize=figsize)
    for i, (ax, model) in enumerate(zip(axes.flatten(), models)):
        ax.set_title(model)
        if equal_aspect:
            ax.set_aspect('equal')
        ax.axhline(0, c='k', lw=0.5, ls='--')
        model_data = pd.concat([x_data[model], y_data[model], sig_data[model]], axis=1, keys=['x', 'y', 'significant'])

        for prompt in model_data.index:
            x_val, y_val, _ = model_data.loc[prompt]
            colour = colours[prompt]
            ax.plot(x_val, y_val, marker='o', c=colour, label=prompt)
            ax.annotate(prompt, (x_val, y_val), textcoords='offset points', xytext=(4, 4), fontsize=8, color=colour)

        for pair in (['GSM', 'NL-simple'], ['NL-simple', 'NL-structured'], ['NL-simple', 'code-simple'],
                     ['NL-structured', 'code-structured'], ['code-simple', 'code-structured']):
            pair_data = model_data.loc[pair]
            ax.plot(pair_data['x'], pair_data['y'], lw=0.2, ls='--', c='darkslategrey')

        model_sig_data = model_data[~model_data.significant.isna()]
        model_sig_data = model_sig_data[model_sig_data.significant]
        if sig_data.size:
            ax.plot(model_sig_data['x'], model_sig_data['y'], marker='o', lw=0, c='none', mec='darkred', ms=12, label=r'significant $\Delta_{var}$')

    for ax in axes[:, 0]:
        ax.set_ylabel("Variant performance delta\n" + r"($\Delta_{var}$), pp")
    for ax in axes[-1, :]:
        ax.set_xlabel("Mean accuracy on GSM-Base, %")


    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Prompt formats & significance of effects', loc='lower center', ncol=6, frameon=True)
    fig.tight_layout(rect=(0, bottom_margin, 1, 1))
    return fig


@save_plot("number_counts")
def plot_number_counts(raw_counts_df: pd.DataFrame, binned_counts_df: pd.DataFrame, cum_cap: float = 100):
    """Plot binned percentage bars of extracted numbers and their cumulative curves.

    Returns a matplotlib Figure.
    """
    plot_bin_positions = np.arange(len(binned_counts_df))
    plot_bin_centers = plot_bin_positions + 0.5
    variants = list(binned_counts_df.columns)
    n_variants = len(variants)

    fig, (ax_count, ax_cum) = plt.subplots(2, 1, figsize=(10, 7))
    total_bar_width = 0.8
    bar_width = total_bar_width / max(n_variants, 1)
    bar_offset = (1.0 - total_bar_width) / 2.0
    for idx, variant_name in enumerate(variants):
        percentages = binned_counts_df[variant_name] / binned_counts_df[variant_name].sum() * 100
        bar_positions = plot_bin_positions + bar_offset + idx * bar_width
        color = VARIANT_COLOURS.get(variant_name, None)
        ax_count.bar(
            bar_positions,
            percentages,
            width=bar_width,
            align='edge',
            alpha=0.8,
            edgecolor='white',
            linewidth=0.8,
            label=variant_name,
            color=color,
        )

        raw_counts = raw_counts_df[variant_name]
        raw_counts = raw_counts[raw_counts > 0]
        cum_perc = raw_counts.cumsum() / raw_counts.sum() * 100
        ax_cum.plot(cum_perc.index, cum_perc,
                    marker='.', ms=4, lw=1.0, label=variant_name, color=color)

    for ax in (ax_count, ax_cum):
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))
        ax.legend(title='Dataset')

    ax_count.set_xticks(plot_bin_centers)
    ax_count.set_xticklabels(binned_counts_df.index)
    ax_count.set_xlabel('Extracted number buckets')
    ax_count.set_ylabel('Percent of variant total')

    ax_cum.set_xlabel('Extracted numbers')
    ax_cum.set_ylabel('Cumulative percent')
    ax_cum.set_xlim(-5, cum_cap + 5)

    fig.tight_layout()

    return fig
