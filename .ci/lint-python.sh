#!/bin/sh

echo "running pre-commit checks"
pre-commit run --all-files || exit 1
echo "done running pre-commit checks"

echo "running mypy"
mypy \
    --config-file=./python-package/pyproject.toml \
    ./python-package \
|| true
echo "done running mypy"
