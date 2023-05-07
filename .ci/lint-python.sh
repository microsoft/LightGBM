#!/bin/sh

echo "running flake8"
flake8 \
    --config=./python-package/setup.cfg \
    --ignore=E501,W503 \
    --exclude=./.nuget,./external_libs,./python-package/build,./python-package/compile \
    . \
|| exit -1
echo "done running flake8"

echo "running pydocstyle"
pydocstyle \
    --config=./python-package/pyproject.toml \
    . \
|| exit -1
echo "done running pydocstyle"

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
