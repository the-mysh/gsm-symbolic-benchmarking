# GSM-Symbolic Benchmarker

Reproducing and extending GSM-Symbolic benchmark


## Installation and related notes
This project's set up to run with a [uv environment](https://docs.astral.sh/uv/).

Project dependencies are included in [pyproject.toml](./pyproject.toml).

To create a virtual environment from pyproject.toml / update your virtual environment from a changed pyproject.toml:
```commandline
$ uv sync
```
(from project root).

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
