#!/bin/bash

set -e -u -o pipefail

rm -rf ./lightgbm-python
cp -R ./python-package ./lightgbm-python

cp -R ./cmake ./lightgbm-python/
cp CMakeLists.txt ./lightgbm-python/
cp -R ./external_libs ./lightgbm-python/
cp -R ./include ./lightgbm-python/
cp LICENSE ./lightgbm-python/
cp -R ./src ./lightgbm-python/
cp -R ./swig ./lightgbm-python/
cp VERSION.txt ./lightgbm-python/
cp -R ./windows ./lightgbm-python/

pushd ./lightgbm-python

pip install .
