# GSM-Symbolic Benchmarker

Reproducing and extending GSM-Symbolic benchmark. 

See [notebooks](./notebooks), in particular [this one](./notebooks/full_results.ipynb), for analysis of results.

Benchmarking script with configurable options available [here](./src/gsm_benchmarker/scripts/benchmark.py).

See [resources](./src/gsm_benchmarker/resources) for definitions of alternative prompt formats used in this study.
Prompt formats usage examples shown [in this notebook](./notebooks/prompting_experiments.ipynb) 
(note: requires an `.env` file containing `HUGGINGFACEHUB_API_TOKEN` and `HF_HOME` present in the execution directory).


## Installation and related notes
This project's set up to run with a [uv environment](https://docs.astral.sh/uv/).

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

**Note:** things might be much easier to set up if you use conda instead of uv (if you have the option). 


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
