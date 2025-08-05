#!/bin/bash

set -e -E -u -o pipefail

echo "installing lightgbm and its dependencies"
pip install \
    --prefer-binary \
    --upgrade \
    -r ./.ci/pip-envs/requirements-oldest.txt \
    dist/*.whl

echo "installed package versions:"
pip freeze

echo ""
echo "checking that examples run without error"

# run a few examples to test that Python-package minimally works
echo ""
echo "--- advanced_example.py ---"
echo ""
python ./examples/python-guide/advanced_example.py || exit 1

echo ""
echo "--- logistic_regression.py ---"
echo ""
python ./examples/python-guide/logistic_regression.py || exit 1

echo ""
echo "--- simple_example.py ---"
echo ""
python ./examples/python-guide/simple_example.py || exit 1

echo ""
echo "--- sklearn_example.py ---"
echo ""
python ./examples/python-guide/sklearn_example.py || exit 1

echo ""
echo "done testing on oldest supported Python version"
