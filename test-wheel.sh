#!/bin/sh

set -e -u -o pipefail

./build-python install
pydistcheck ./dist/*.whl

python -c "import lightgbm"
echo "success!"
sleep 2
pytest tests/python_package_test/test_basic.py
