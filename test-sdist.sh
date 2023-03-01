#!/bin/sh

set -e -u -o pipefail

./build-python.sh sdist
pydistcheck --inspect ./dist/*.tar.gz
pip uninstall --yes lightgbm
MAKEFLAGS="-j4" \
pip install -v ./dist/*.tar.gz
python -c "import lightgbm"
echo "success!"
sleep 2
pytest tests/python_package_test/test_basic.py
