#!/bin/bash

echo 204

export SKBUILD_LOGGING_LEVEL="INFO"

pip --version
pip install pytest numpy pandas scipy scikit-learn psutil cloudpickle


git clone --recursive -b docs/install-py https://github.com/microsoft/LightGBM.git
cd LightGBM
sh ./build-python.sh install --bit32


tests="$BUILD_SOURCESDIRECTORY/tests/python_package_test"

pytest $tests
