# Experimentation

A _Python_ service to execute multiple experiments with given llms build with poetry.

## Prerequisites

* Python == 3.9, you can use for example [pyenv](https://github.com/pyenv/pyenv#installation) to manage that
* [Poetry](https://python-poetry.org/docs/#installation)

## Installing dependencies

```bash
make install
```

## Running it in a terminal

```bash
poetry run python -m experimentation.presentation.cli --help
```

or directly with python

```bash
python -m experimentation.presentation.cli --help
```

for example:

```bash
NUMEXPR_MAX_THREADS=56 MLFLOW_TRACKING_URI="http://localhost:5000" python -m experimentation.presentation.cli --experiment-path ../datalake/experiments/llama2/zero-shot-english/ --dataset-path ../datalake/ --model-path ../datalake/models/llama-2-7b.Q5_K_M.gguf --use-mlflow true
```

## Tests and checks

To run all tests and checks:

```bash
make check
```

To run all tests (unit and integration):

```bash
make test
```

### unit-tests

To just run unit-tests:

```bash
make unit-test
```

### integration-tests

To just run integration-tests:

```bash
make integration-test
```

### Auto-formatting

```bash
make auto-format
```

### Linting

```bash
make lint
```

### Check types

```bash
make type-check
```
