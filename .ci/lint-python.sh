#!/bin/sh

echo "running flake8"
flake8 \
    --ignore=E501,W503 \
    --exclude=./.nuget,./external_libs,./python-package/build,./python-package/compile \
    "${DIR_TO_CHECK}" \
|| exit -1
echo "done running flake8"

echo "running pydocstyle"
pydocstyle \
    "${DIR_TO_CHECK}" \
|| exit -1
echo "done running pydocstyle"

echo "running isort"
isort \
    --check-only \
    "${DIR_TO_CHECK}" \
|| exit -1
echo "done running isort"

echo "running mypy"
mypy \
    "${DIR_TO_CHECK}/python-package" \
|| true
echo "done running mypy"
