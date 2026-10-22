# The Importance of Being Statistically Earnest

This repository contains code, prompt formats, and other metadata used in the paper: 
[The Importance of Being Statistically Earnest: A Critical Re-evaluation of GSM-Symbolic](https://arxiv.org/abs/2605.28700),
presented at [EMNLP 2026](https://2026.emnlp.org/) (main conference; track: _Resources and Evaluation_).

**TL;DR**: Scientist from Apple [showed](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ec2e7a896f8250986b3907f57621ce94-Abstract-Conference.html)
that LLMs cannot solve grade school math problems if the openly known questions are modified (by changing names or values), pointing to data contamination.
They took this result to mean that _LLMs cannot genuinely reason_.
We show that:
1) 60\% of their results (within the _GSM-Symbolic main_ study) are not statistically significant,
2) if number load (using higher numerical values) is accounted for, that proportion rises to 80\%,
3) models display varied failure modes when alternative prompt formats are used, so blanket statements about LLM reasoning capabilities should be avoided.

---


See [notebooks](./notebooks), in particular [this one](./notebooks/full_results.ipynb), for analysis of results.

Benchmarking script with configurable options available [here](./src/gsm_benchmarker/scripts/benchmark.py).

See [resources](./src/gsm_benchmarker/resources) for definitions of alternative prompt formats used in this study.
Prompt formats usage examples shown [in this notebook](./notebooks/prompting_experiments.ipynb) 
(note: requires an `.env` file containing `HUGGINGFACEHUB_API_TOKEN` and `HF_HOME` present in the execution directory).


## Installation and related notes
This project's set up to run with a [uv environment](https://docs.astral.sh/uv/) (see also the [cheat sheet](#uv-cheat-sheet) below)..

Project dependencies are included in [pyproject.toml](./pyproject.toml).

To create a virtual environment from pyproject.toml / update your virtual environment from a changed pyproject.toml:
```commandline
$ uv sync
```
(from project root).

Note [uv dependency caching policy](https://docs.astral.sh/uv/concepts/cache/#dependency-caching).

The `pymer4` package, needed for statistical significance assessment with GLMM, depends on R with a few libraries.
For the specific use case of R version 3.6.3 (which is all I have on my shared system), run the [R setup script](./setup.R):
```commandline
$ Rscript setup.R
```
**Important:** Whenever the installer prompts you to choose whether to update installed packages - ignore the updates 
(enter empty line).

_Note: I do not plan on maintaining the pymer4 dependency and I do not expect to be able to debug issues with it;
if time allows, I will swap the GLMM backend to [statsmodels](https://www.statsmodels.org/stable/mixed_glm.html)._

### `uv` cheat sheet

To install the current package in the environment:

```commandline
$ uv pip install --no-deps --no-build-isolation -e .
```

To install a new package and add the dependency to pyproject.toml:
```commandline
$ uv add <package>
``` 
Note: this usually pins the version in pyproject.toml. You might want to edit the pyproject.toml afterwards to loosen the requirement (depending on the package being installed).

Activating venv in command line:
- Windows: `.venv/Scripts/activate`
- Linux: `source .venv/bin/activate`

Running jupyter notebook:
```commandline
$ uv run jupyter notebook
```

Running Python from uv venv:
```commandline
$ uv run python
```

Running any of the scripts from uv venv: 
```commandline
$ uv run <path-to-script>
```
(doesn't need venv activated; just running this from root works)
