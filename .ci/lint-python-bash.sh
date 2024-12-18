#!/bin/bash

set -e -E -u -o pipefail

echo "running pre-commit checks"
SKIP=yamllint pre-commit run --all-files || exit 1
echo "done running pre-commit checks"

echo "running mypy"
mypy \
    --config-file=./python-package/pyproject.toml \
    ./python-package \
|| true
echo "done running mypy"
