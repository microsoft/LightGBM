#!/bin/bash

echo 207

export SKBUILD_LOGGING_LEVEL="INFO"

pip --version
pip install pytest numpy pandas scipy scikit-learn psutil cloudpickle


git clone --recursive -b docs/install-py https://github.com/microsoft/LightGBM.git
cd LightGBM
MSBuild.exe windows/LightGBM.sln /p:Configuration=DLL /p:Platform=x64 /p:PlatformToolset=v140
sh ./build-python.sh install --precompile


tests="$BUILD_SOURCESDIRECTORY/tests/python_package_test"

pytest $tests
