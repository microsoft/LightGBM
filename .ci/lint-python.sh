#!/bin/sh

echo "running pre-commit checks"
pre-commit run --all-files || exit 1
echo "done running pre-commit checks"

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
