#!/bin/bash

set -e -u -o pipefail

pip uninstall -y lightgbm

rm -rf \
    ./lightgbm-python \
    ./lib_lightgbm.so \
    ./lightgbm \
    ./python-package/build \
    ./python-package/build_cpp \
    ./python-package/compile \
    ./python-package/dist \
    ./python-package/lightgbm.egg-info

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

python -m build .
pydistcheck dist/* --inspect
pip install dist/*.whl
python -c "import lightgbm"
