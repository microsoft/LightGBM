#!/bin/sh

DIR_TO_CHECK=${1}

echo "running flake8"
flake8 \
    --ignore=E501,W503 \
    --exclude=./.nuget,./external_libs,./python-package/build,./python-package/compile \
    "${DIR_TO_CHECK}" \
|| exit -1
echo "done running flake8"

echo "running pydocstyle"
pydocstyle \
    --convention=numpy \
    --add-ignore=D105 \
    --match-dir="^(?!^external_libs|test|example).*" \
    --match="(?!^test_|setup).*\.py" \
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
    --ignore-missing-imports \
    --exclude 'build/' \
    --exclude 'compile/' \
    --exclude 'docs/' \
    --exclude 'examples/' \
    --exclude 'external_libs/' \
    --exclude 'tests/' \
    "${DIR_TO_CHECK}/python-package" \
|| true
echo "done running mypy"
