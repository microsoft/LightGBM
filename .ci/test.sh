#!/bin/bash

echo 109

export SKBUILD_LOGGING_LEVEL="INFO"

pip --version
pip install pytest numpy pandas scipy scikit-learn psutil cloudpickle


# git clone --recursive -b docs/install-py https://github.com/microsoft/LightGBM.git
# cd LightGBM
# sh ./build-python.sh install --mingw
pip install lightgbm -v --no-binary lightgbm --config-settings=cmake.args="-AWin32" --config-setting=cmake.define.__BUILD_FOR_PYTHON=ON


tests="$BUILD_SOURCESDIRECTORY/tests/python_package_test"

pytest $tests
