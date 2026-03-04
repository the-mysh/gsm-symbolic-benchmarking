import os
import logging
from functools import cached_property
import pandas as pd
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np

from gsm_benchmarker.results_analyser.model import ModelResultsAnalyser


logger = logging.getLogger(__name__)


class MultiModelResultsAnalyser:
    def __init__(self, dir_path: str | Path, load_full_data: bool = False):
        self._dir_path = Path(dir_path).resolve()

        summary_data_dict, full_data_dict = self._load_data(self._dir_path, load_full_data=load_full_data)
        self._summary_data = self._make_summary_df(summary_data_dict)
        self._full_data = self._make_full_df(full_data_dict) if full_data_dict else None

    @cached_property
    def full_data(self) -> pd.DataFrame:
        if self._full_data is None:
            self._full_data = self._load_full_data()
        return self._full_data

    @property
    def summary_data(self) -> pd.DataFrame:
        return self._summary_data

    @staticmethod
    def _load_data(dir_path: Path, load_full_data: bool = False):
        full_data_dict = {}
        summary_data_dict = {}

        logger.debug("Loading per-model results")
        for item_name in os.listdir(dir_path):
            item_path = dir_path / item_name
            if item_path.is_dir():
                logger.warning(f"The algorithm is not meant for non-flat directories; found subfolder '{item_name}'")
                continue
            model_results = ModelResultsAnalyser(item_path)
            model_name = ''.join(item_name.split('.')[:-1])

            parts = model_name.split('_')
            family_name = parts[0]
            model_name = '.'.join(parts[1:])

            if load_full_data:
                full_data_dict[model_name] = model_results.data
            s = model_results.get_total_accuracy_and_std()
            s_strict = model_results.get_total_accuracy_and_std(strict=True)
            summary_data_dict[model_name] = {'accuracy': s[0], 'std': s[1],
                                             'strict_accuracy': s_strict[0], 'strict_std': s_strict[1],
                                             'family': family_name}

        return summary_data_dict, full_data_dict

    def get_core_stats(self):
        return self.full_data.groupby(['model', 'instance'])[['correct', 'correct_strict', 'babbling']].mean()

    def get_accuracies_per_model_and_template_id(self):
        res = self.full_data.groupby(['model', 'id'])[['correct', 'correct_strict']].mean()
        res = pd.concat(
            [a.rename('accuracy') for a in (res.correct, res.correct_strict)],
            keys=['standard', 'discounted'],
            names=('metric', 'model', 'id')
        ).swaplevel(0, 1).swaplevel(1, 2).sort_index()
        return res

    @staticmethod
    def _make_summary_df(summary_data_dict):
        data_df = pd.DataFrame(summary_data_dict)
        return data_df.T

    @staticmethod
    def _make_full_df(full_data_dict):
        df = pd.concat(full_data_dict.values(), keys=full_data_dict.keys(), names=['model', 'old_index'])
        df = df.reset_index().drop('old_index', axis=1)
        return df

    def _load_full_data(self):
        _, data_dict = self._load_data(dir_path=self._dir_path, load_full_data=True)
        return self._make_full_df(data_dict)

    @property
    def models(self) -> list[str]:
        return self.full_data.model.unique().tolist()

    @property
    def instances(self) -> list[int]:
        return self.full_data.instance.unique().tolist()

    @property
    def ids(self) -> list[int]:
        return self.full_data.id.unique().tolist()

    def filter(self, **pairs: Any) -> pd.DataFrame:
        df = self.full_data
        for (column, value) in pairs.items():
            df = df[df[column] == value]
        return df

    def get_example(self, id: int, instance: int, model: str) -> dict[str, Any] | None:
        df = self.filter(id=id, instance=instance, model=model)

        if not len(df):
            if model not in self.models:
                raise ValueError(f"Model {model} does not exist in data")
            if id not in self.ids:
                raise ValueError(f"Id {id} does not exist in data")
            if instance not in self.instances:
                raise ValueError(f"Instance {instance} does not exist in data")

            # each exists, just not the combo
            logger.warning(f"No example with template id {id}, instance number {instance},"
                           f"and model {model} found")
            return None

        if len(df) > 1:
            raise RuntimeError(f"Multiple examples with the same template id {id}, "
                               f"instance number {instance}, and model {model} found")

        return df.to_dict(orient='index')[df.index[0]]

    def get_babbler_counts(self) -> pd.DataFrame:
        babbler_examples = self.full_data[self.full_data.babbling]

        babbler_counts = babbler_examples["model"].value_counts()
        babbler_counts.name = "babbler count"

        babbler_percentage = babbler_counts / self.full_data["model"].value_counts()
        babbler_percentage.name = "babbler percentage"

        b = pd.concat((self.summary_data, babbler_counts, babbler_percentage), axis=1)
        b.fillna(0, inplace=True)
        b.sort_values(['babbler percentage', 'accuracy'], ascending=False)

        return b

    def plot_babblers_by_family(self, b: pd.DataFrame | None = None, strict: bool = False):
        if b is None:
            b = self.get_babbler_counts()

        acc_column = 'strict_accuracy' if strict else 'accuracy'

        fig, ax = plt.subplots()
        ax.set_ylabel(f"{acc_column.replace('_', ' ')}, %")
        ax.set_xlabel("babbler factor, %")
        for family in b.family.unique():
            bb =  b[b.family == family]
            ax.scatter(100*bb['babbler percentage'], 100*bb[acc_column], marker='d', label=family)
        ax.legend(fancybox=True, framealpha=0.5, frameon=True, title='Family')
        ax.set_aspect('equal')

        m = 1
        lims = (-m, 100+m)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        return fig

    def compare_babblers(self, other: "MultiModelResultsAnalyser", title1: str, title2: str, b1=None, b2=None):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        if b1 is None:
            b1 = self.get_babbler_counts()

        if b2 is None:
            b2 = other.get_babbler_counts()

        for i, c in enumerate(('accuracy', 'strict_accuracy', 'babbler percentage')):
            ax = axes[i]
            ax.set_title(f"{c.replace('_', ' ').capitalize()}, %")

            for family in b2.family.unique():
                bb2 =  b2[b2['family'] == family]
                bb1 = b1[b1['family'] == family]
                bb1 = bb1[bb1.index.isin(bb2.index)]
                ax.scatter(100*bb1[c], 100*bb2[c], marker='d', label=family)

            m = 1
            lims = (-m, 100+m)
            ax.set_xlim(lims)
            ax.set_ylim(lims)

            ax.set_aspect('equal')
            ax.axline([0, 0], [1, 1], c='k', lw=1, linestyle='--')
            ax.legend(fancybox=True, framealpha=0.5, frameon=True, title='Family')
            ax.set_xlabel(title1)
            ax.set_ylabel(title2)

    @staticmethod
    def _plot_bars(counts_df: pd.DataFrame, color=None, title: str = None, legend_title: str | None = None,
                   category_name: str | None = None, rotate_labels: bool = True, percentage: bool = False):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [3, 1]})

        counts_df.plot(
            kind='bar',
            stacked=True,
            ax=axes[0],
            color=color,
            legend=False
        )
        if category_name:
            axes[0].set_title(f"By {category_name}")
            axes[0].set_xlabel(category_name.capitalize())
        axes[0].set_ylabel('% all answers' if percentage else 'Count')
        axes[0].legend(title=legend_title, fancybox=True, framealpha=0.7, frameon=True)

        axes[0].tick_params(axis='x', rotation=45 if rotate_labels else 0)

        if rotate_labels:
            for label in axes[0].get_xticklabels():
                label.set_ha('right')

        counts_df.sum().plot(
            kind='pie',
            ax=axes[1],
            colors=color,
            textprops={'size': 'smaller'},
        )
        axes[1].set_title("Combined")

        if title is not None:
            fig.suptitle(title)

        fig.tight_layout() # Adjust layout to prevent labels from being cut off

        return fig

    def plot_result_class_by_model(self, title: str | None = None):

        counts_df = self._full_data.groupby(['model', 'result_class']).size().unstack(fill_value=0)
        counts_df = counts_df.reindex(columns=['CORRECT', 'BABBLING', 'INCORRECT', 'FAILED'], fill_value=0)

        fig = self._plot_bars(
            counts_df,
            color=['green', '#b8bd39', '#d15f26', 'saddlebrown'],
            title=title or "Result class",
            category_name="model"
        )

        return fig

    def get_failed_answer_cases(self):
        return self._full_data[self.full_data.predicted_numerical_result.isna()]

    def plot_error_types_by_model(self, title: str | None = None, percentage: bool = False):

        failed = self.get_failed_answer_cases()

        counts_df = failed.groupby(['model', 'detected_result_pattern']).size().unstack(fill_value=0)

        if percentage:
            counts_df = self._get_percentages(counts_df, 'model')

        fig = self._plot_bars(
            counts_df,
            title=title or "Error types",
            category_name="model",
            percentage=percentage
        )

        return fig

    def _get_percentages(self, counts_df, index_col: str):
        full_counts = self.full_data.groupby([index_col]).size()
        full_counts = full_counts.reindex(counts_df.index, fill_value=0)
        counts_df = counts_df / np.repeat(np.atleast_2d(full_counts.to_numpy()), counts_df.shape[1], axis=0).T * 100

        return counts_df

    def plot_error_types_by_question_id(self, title: str | None = None, max_questions: int | None = None,
                                        highest: bool = True, percentage=False):
        failed = self.get_failed_answer_cases()

        counts_df = failed.groupby(['id', 'detected_result_pattern']).size().unstack(fill_value=0)
        counts_df = counts_df.reindex(counts_df.sum(axis=1).sort_values(ascending=False).index)

        if max_questions and len(counts_df) > max_questions:
            if highest:
                counts_df = counts_df[:max_questions]
                counts_df = counts_df.reindex(counts_df.index.tolist() + ["..."], fill_value=0)
            else:
                counts_df = counts_df[-max_questions:]
                counts_df = counts_df.reindex(["..."] + counts_df.index.tolist(), fill_value=0)

        if percentage:
            counts_df = self._get_percentages(counts_df, 'id')

        fig = self._plot_bars(
            counts_df,
            title=title or "Error types",
            category_name="question template id",
            rotate_labels=False,
            percentage=percentage
        )

        return fig
