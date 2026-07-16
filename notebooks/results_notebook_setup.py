import logging
from functools import cached_property
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd
import json


from gsm_benchmarker.results_analyser.prompt_result import PromptResult
from gsm_benchmarker.results_analyser.plotting_utils import Colour

logger = logging.getLogger('notebook')


plt.style.use('default')
plt.style.use('seaborn-v0_8-muted')
plt.style.use('seaborn-v0_8-darkgrid')


METRIC = "correct"
ALPHA = 0.05


here = Path(__file__).parent.resolve()
RESULTS_ROOT = (here.parent.parent.parent/"data/gsm-symbolic/outputs").resolve()

RESULTS_FOLDERS = {
    'gsm': "noq_default_full__12_05/final",
    'nonformal': "noq_nonformalised__12_05/final",
    'formal': "noq_formalised__12_05/final",
    'short_code': "noq_code_short__12_05/final",
    'long_code': "noq_code_long__12_05/final"
}

RESOURCES = here/"resources"

OUTPUTS = here/"outputs"
os.makedirs(OUTPUTS, exist_ok=True)
OUTPUTS_FOLDER = str(OUTPUTS) + "/"


class _ResultsLoader:
    def __init__(self, res_root, res_folders, metric, save_dest=None):
        self.results_root = res_root
        self.results_folders = res_folders
        self.result_kwargs = dict(metric=metric, save_dest=save_dest)

    @cached_property
    def gsm(self):
        return PromptResult(
            self.results_root / self.results_folders['gsm'],
            colour=Colour('green'),
            full_label="GSM prompt",
            **self.result_kwargs
        )

    @cached_property
    def nonformal(self):
        return PromptResult(
            self.results_root / self.results_folders['nonformal'],
            colour=Colour("skyblue"),
            full_label="Simple NL prompt",
            short_label="NL-simple",
            baseline=self.gsm.mres,
            **self.result_kwargs
        )

    @cached_property
    def formal(self):
        return PromptResult(
            self.results_root / self.results_folders['formal'],
            colour=Colour("steelblue"),
            full_label="Structured NL prompt",
            short_label="NL-structured",
            baseline=self.gsm.mres,
            **self.result_kwargs
        )

    @cached_property
    def short_code(self):
        return PromptResult(
            self.results_root / self.results_folders['short_code'],
            colour=Colour("mediumpurple").lighten(0.2),
            full_label="Simple code-output prompt",
            short_label="code-simple",
            baseline=self.gsm.mres,
            **self.result_kwargs
        )

    @cached_property
    def long_code(self):
        return PromptResult(
            self.results_root / self.results_folders['long_code'],
            colour=Colour("rebeccapurple"),
            full_label="Structured code-output prompt",
            short_label="code-structured",
            baseline=self.gsm.mres,
            **self.result_kwargs
        )

    @cached_property
    def full_results(self):
        return {
            'GSM': self.gsm,
            'Simple NL': self.nonformal,
            'Structured NL': self.formal,
            'Simple Code': self.short_code,
            'Structured Code': self.long_code
        }


results_loader = _ResultsLoader(RESULTS_ROOT, RESULTS_FOLDERS, METRIC, save_dest=OUTPUTS)


with open(RESOURCES/'mirzadeh-data.json') as f:
    original_results_df = pd.DataFrame(json.load(f))
    original_results_df = original_results_df.set_index('model')

    models = results_loader.gsm.mres.models
    original_results_df = original_results_df[original_results_df.index.isin(models)]

    original_model_order = original_results_df.index.tolist()


significant_models_path = RESOURCES/"significant_models.txt"
if significant_models_path.exists():
    with open(significant_models_path, "r") as f:
        significant_models = [m.rstrip('\n') for m in f.readlines()]
else:
    significant_models = None
