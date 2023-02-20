#!/bin/sh

set -e -u -o pipefail

pip uninstall --yes lightgbm
pip install -v ./lightgbm-python/dist/*.tar.gz
python -c "import lightgbm"
echo "success!"
sleep 2
pytest tests/python_package_test/test_basic.py
