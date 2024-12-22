#!/bin/bash

echo 45

export SKBUILD_LOGGING_LEVEL="INFO"

pip --version
pip install pytest numpy pandas scipy scikit-learn psutil cloudpickle



sh ./build-python.sh install


tests="$BUILD_SOURCESDIRECTORY/tests/python_package_test"

pytest $tests
