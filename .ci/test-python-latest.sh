#!/bin/bash

set -e -E -u -o pipefail

# latest versions of lightgbm's dependencies,
# including pre-releases and nightlies
#
# ref: https://github.com/pydata/xarray/blob/31111b3afe44fd6f7dac363264e94186cc5168d2/.github/workflows/upstream-dev-ci.yaml
echo "installing testing dependencies"
python -m pip install \
    cloudpickle \
    psutil \
    pytest
echo "done installing testing dependencies"

echo "installing lightgbm's dependencies"
python -m pip install \
    --extra-index-url https://pypi.anaconda.org/scientific-python-nightly-wheels/simple \
    --prefer-binary \
    --pre \
    --upgrade \
        'numpy>=2.0.0.dev0' \
        'matplotlib>=3.10.0.dev0' \
        'pandas>=3.0.0.dev0' \
        'scikit-learn>=1.6.dev0' \
        'scipy>=1.15.0.dev0'

python -m pip install \
    --extra-index-url https://pypi.fury.io/arrow-nightlies/ \
    --prefer-binary \
    --pre \
    --upgrade \
        'pyarrow>=17.0.0.dev0'

python -m pip install \
    'cffi>=1.15.1'

echo "done installing lightgbm's dependencies"

echo "installing lightgbm"
pip install --no-deps dist/*.whl
echo "done installing lightgbm"

echo "installed package versions:"
pip freeze

echo ""
echo "running tests"
pytest tests/c_api_test/
pytest tests/python_package_test/
