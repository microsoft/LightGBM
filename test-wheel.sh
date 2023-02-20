#!/bin/sh

set -e -u -o pipefail

pydistcheck ./dist/*.whl
pip uninstall --yes lightgbm
pip install ./dist/*.whl
python -c "import lightgbm"
echo "success!"
sleep 2
pytest tests/python_package_test/test_basic.py
