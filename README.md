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

To install a new package and 
```commandline
$ uv add <package>
``` 
installs the package and adds the dependency to pyproject.toml (usually pinning the version).

Activating venv in command line:
- Windows: `.venv/Scripts/activate`
- Linux: `source .venv/bin/activate`

Running jupyter notebook:
```commandline
$ uv run --with jupyter jupyter notebook
```