#!/bin/bash

echo 45

export SKBUILD_LOGGING_LEVEL="INFO"

pip --version
pip install pytest numpy pandas scipy scikit-learn psutil cloudpickle



git clone --recursive https://github.com/microsoft/LightGBM.git
cd LightGBM
# export CXX=g++-14 CC=gcc-14  # macOS users, if you decided to compile with gcc, don't forget to specify compilers
sh ./build-python.sh install


tests="$BUILD_SOURCESDIRECTORY/tests/python_package_test"

pytest $tests
