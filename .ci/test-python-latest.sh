#!/bin/bash

set -e -E -u -o pipefail

# oldest versions of dependencies published after
# minimum supported Python version's first release,
# for which there are wheels compatible with the
# python:{version} image
#
# see https://devguide.python.org/versions/
#
echo "installing testing dependencies"
python -m pip install \
    cloudpickle \
    psutil \
    pytest
echo "done installing testing dependencies"

# ref: https://github.com/pydata/xarray/blob/31111b3afe44fd6f7dac363264e94186cc5168d2/.github/workflows/upstream-dev-ci.yaml
echo "installing lightgbm's dependencies"
python -m pip install \
    --extra-index-url https://pypi.anaconda.org/scientific-python-nightly-wheels/simple \
    --prefer-binary \
    --pre \
    --upgrade \
        numpy \
        scipy \
        matplotlib \
        pandas

python -m pip install \
    --extra-index-url https://pypi.fury.io/arrow-nightlies/ \
    --prefer-binary \
    --pre \
    --upgrade \
    pyarrow

echo "done installing lightgbm's dependencies"

echo "installing lightgbm"
pip install --no-deps dist/*.whl || exit 1
echo "done installing lightgbm"

echo "installed package versions:"
pip freeze

echo ""
echo "running tests"
pytest tests/python_package_test/test_basic.py
pytest tests/python_package_test/test_engine.py
pytest tests/python_package_test/test_sklearn.py
