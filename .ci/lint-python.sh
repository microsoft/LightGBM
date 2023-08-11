#!/bin/sh

echo "running ruff"
ruff check \
    --config=./python-package/pyproject.toml \
    . \
|| exit -1
echo "done running ruff"

echo "running isort"
isort \
    --check-only \
    --settings-path=./python-package/pyproject.toml \
    . \
|| exit -1
echo "done running isort"

echo "running mypy"
mypy \
    --config-file=./python-package/pyproject.toml \
    ./python-package \
|| true
echo "done running mypy"
