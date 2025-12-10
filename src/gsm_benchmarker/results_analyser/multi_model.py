import os
import logging
from functools import cached_property
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Any
import matplotlib.pyplot as plt

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
        for item_name in tqdm(os.listdir(dir_path), desc="Model"):
            item_path = dir_path / item_name
            if item_path.is_dir():
                logger.warning(f"The algorithm is not meant for non-flat directories; found subfolder '{item_name}'")
                continue
            model_results = ModelResultsAnalyser(item_path)
            model_name = ''.join(item_name.split('.')[:-1])

            if load_full_data:
                full_data_dict[model_name] = model_results.data
            s = model_results.get_total_accuracy_and_std()
            s_strict = model_results.get_total_accuracy_and_std(strict=True)
            summary_data_dict[model_name] = {'accuracy': s[0], 'std': s[1],
                                             'strict_accuracy': s_strict[0], 'strict_std': s_strict[1]}

        return summary_data_dict, full_data_dict

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

        family = self.summary_data.index.to_series().apply(lambda v: v.split('_')[0])
        family.name = 'family'

        b = pd.concat((family, self.summary_data, babbler_counts, babbler_percentage), axis=1)
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
        for family in b['family'].unique():
            bb =  b[b['family'] == family]
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

    def plot_result_class_by_model(self, title: str | None = None):
        fig, ax = plt.subplots(figsize=(12, 6))

        counts_df = self._full_data.groupby(['model', 'result_class']).size().unstack(fill_value=0)
        counts_df = counts_df.reindex(columns=['CORRECT', 'BABBLING', 'INCORRECT', 'FAILED'], fill_value=0)
        counts_df.index = ['_'.join(m.split('_')[1:]) for m in counts_df.index]

        counts_df.plot(
            kind='bar',
            stacked=True,
            ax=ax,
            color=['green', '#b8bd39', '#d15f26', 'saddlebrown'] # Optional: set custom colors
        )

        ax.set_title(title if title is not None else 'Result Class Counts by Model')
        ax.set_xlabel('Model')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45) # Rotate x-axis labels for better readability if model names are long
        ax.legend(title='Result', fancybox=True, framealpha=0.8, loc='lower right', frameon=True)

        fig.tight_layout() # Adjust layout to prevent labels from being cut off

        for label in ax.get_xticklabels():
            label.set_ha('right')

        return fig
