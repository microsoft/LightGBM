#!/bin/bash

set -e -E -u -o pipefail

# oldest versions of dependencies published after
# minimum supported Python version's first release
#
# see https://devguide.python.org/versions/
#
echo "installing lightgbm's dependencies"
pip install \
  'cffi==1.15.1' \
  'dataclasses' \
  'numpy==1.16.6' \
  'pandas==0.24.0' \
  'pyarrow==6.0.1' \
  'scikit-learn==0.18.2' \
  'scipy==0.19.0' \
|| exit 1
echo "done installing lightgbm's dependencies"

echo "installing lightgbm"
pip install --no-deps dist/*.whl || exit 1
echo "done installing lightgbm"

echo "installed package versions:"
pip freeze

echo ""
echo "checking that examples run without error"

# run a few examples to test that Python package minimally works
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
