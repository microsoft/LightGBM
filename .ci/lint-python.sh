#!/bin/bash

set -e -E -u -o pipefail

# this can be re-enabled when this is fixed:
# https://github.com/tox-dev/filelock/issues/337
# echo "running pre-commit checks"
# pre-commit run --all-files || exit 1
# echo "done running pre-commit checks"

echo "running mypy"
mypy \
    --config-file=./python-package/pyproject.toml \
    ./python-package \
|| true
echo "done running mypy"
