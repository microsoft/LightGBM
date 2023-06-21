#!/bin/bash

# downgrade to oldest versions of dependencies published after
# minimum supported Python version's first release
echo "installing lightgbm's dependencies"
pip install \
  'numpy==1.12.0' \
  'pandas==0.19.2' \
  'scikit-learn==0.18.2' \
  'scipy==0.19.0' \
|| exit -1
echo "done installing lightgbm's dependencies"

echo "installing lightgbm"
pip install dist/*.whl || exit -1
echo "done installing lightgbm"

echo "installed package versions:"
pip freeze

echo ""
echo "checking that examples run without error"

# run a few examples to test that Python package minimally works
echo "--- simple_example.py"
#python ./examples/python-guide/simple_example.py || exit -1

echo "--- sklearn_example.py"
python ./examples/python-guide/sklearn_example.py || exit -1

echo "--- advanced_example.py"
python ./examples/python-guide/advanced_example.py || exit -1

echo "done testing on oldest supported Python version"
