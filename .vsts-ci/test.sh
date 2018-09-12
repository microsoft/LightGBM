#!/bin/bash

if [[ $AGENT_OS == "Linux" ]] && [[ $COMPILER == "clang" ]]; then
    export CXX=clang++
    export CC=clang
fi

cd $BUILD_SOURCESDIRECTORY

if [[ $TASK == "pylint" ]]; then
    conda install -y -n $CONDA_ENV pycodestyle
    pycodestyle --ignore=E501,W503 --exclude=./compute,./docs,./.nuget . || exit -1
    exit 0
fi

if [[ $TASK == "if-else" ]]; then
    conda install -y -n $CONDA_ENV numpy
    mkdir $BUILD_SOURCESDIRECTORY/build && cd $BUILD_SOURCESDIRECTORY/build && cmake .. && make lightgbm || exit -1
    cd $BUILD_SOURCESDIRECTORY/tests/cpp_test && ../../lightgbm config=train.conf convert_model_language=cpp convert_model=../../src/boosting/gbdt_prediction.cpp && ../../lightgbm config=predict.conf output_result=origin.pred || exit -1
    cd $BUILD_SOURCESDIRECTORY/build && make lightgbm || exit -1
    cd $BUILD_SOURCESDIRECTORY/tests/cpp_test && ../../lightgbm config=predict.conf output_result=ifelse.pred && python test.py || exit -1
    exit 0
fi

conda install -q -y -n $CONDA_ENV numpy nose scipy scikit-learn pandas matplotlib python-graphviz pytest

if [[ $AGENT_OS == "Darwin" ]] ; then
    ln -sf `ls -d "$(brew --cellar libomp)"/*/lib`/* $CONDA_PREFIX/lib || exit -1  # fix "OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized." (OpenMP library conflict due to conda's MKL)
fi

if [[ $TASK == "sdist" ]]; then
    cd $BUILD_SOURCESDIRECTORY/python-package && python setup.py sdist || exit -1
    pip install $BUILD_SOURCESDIRECTORY/python-package/dist/lightgbm-$LGB_VER.tar.gz -v || exit -1
    cp $BUILD_SOURCESDIRECTORY/python-package/dist/lightgbm-$LGB_VER.tar.gz $BUILD_ARTIFACTSTAGINGDIRECTORY
    pytest $BUILD_SOURCESDIRECTORY/tests/python_package_test || exit -1
    exit 0
elif [[ $TASK == "bdist" ]]; then
    if [[ $AGENT_OS == "Darwin" ]]; then
        cd ${BUILD_REPOSITORY_LOCALPATH}/python-package && python setup.py bdist_wheel --plat-name=macdarwin --universal || exit -1
        cp dist/lightgbm-$LGB_VER-py2.py3-none-macdarwin.whl ${BUILD_ARTIFACTSTAGINGDIRECTORY}/lightgbm-$LGB_VER-py2.py3-none-macosx_10_6_x86_64.macosx_10_7_x86_64.macosx_10_8_x86_64.macosx_10_9_x86_64.macosx_10_10_x86_64.macosx_10_11_x86_64.macosx_10_12_x86_64.macosx_10_13_x86_64.whl
        mv dist/lightgbm-$LGB_VER-py2.py3-none-macdarwin.whl dist/lightgbm-$LGB_VER-py2.py3-none-macosx_10_6_x86_64.macosx_10_7_x86_64.macosx_10_8_x86_64.macosx_10_9_x86_64.macosx_10_10_x86_64.macosx_10_11_x86_64.macosx_10_12_x86_64.macosx_10_13_x86_64.whl
    else
        cd $BUILD_SOURCESDIRECTORY/python-package && python setup.py bdist_wheel --plat-name=manylinux1_x86_64 --universal || exit -1
        cp dist/lightgbm-$LGB_VER-py2.py3-none-manylinux1_x86_64.whl $BUILD_ARTIFACTSTAGINGDIRECTORY
    fi
    pip install $BUILD_SOURCESDIRECTORY/python-package/dist/*.whl || exit -1
    pytest $BUILD_SOURCESDIRECTORY/tests/python_package_test || exit -1
    exit 0
fi

if [[ $TASK == "gpu" ]]; then
    sed -i'.bak' 's/std::string device_type = "cpu";/std::string device_type = "gpu";/' $BUILD_SOURCESDIRECTORY/include/LightGBM/config.h
    grep -q 'std::string device_type = "gpu"' $BUILD_SOURCESDIRECTORY/include/LightGBM/config.h || exit -1  # make sure that changes were really done
    if [[ $METHOD == "pip" ]]; then
        cd $BUILD_SOURCESDIRECTORY/python-package && python setup.py sdist || exit -1
        pip install $BUILD_SOURCESDIRECTORY/python-package/dist/lightgbm-$LGB_VER.tar.gz -v --install-option=--gpu --install-option="--opencl-include-dir=$AMDAPPSDK_PATH/include/" || exit -1
        pytest $BUILD_SOURCESDIRECTORY/tests/python_package_test || exit -1
        exit 0
    fi
fi

mkdir $BUILD_SOURCESDIRECTORY/build && cd $BUILD_SOURCESDIRECTORY/build

if [[ $TASK == "mpi" ]]; then
    if [[ $METHOD == "pip" ]]; then
        cd $BUILD_SOURCESDIRECTORY/python-package && python setup.py sdist || exit -1
        pip install $BUILD_SOURCESDIRECTORY/python-package/dist/lightgbm-$LGB_VER.tar.gz -v --install-option=--mpi || exit -1
        pytest $BUILD_SOURCESDIRECTORY/tests/python_package_test || exit -1
        exit 0
    fi
    cmake -DUSE_MPI=ON ..
elif [[ $TASK == "gpu" ]]; then
    cmake -DUSE_GPU=ON -DOpenCL_INCLUDE_DIR=$AMDAPPSDK_PATH/include/ ..
else
    cmake ..
fi

make _lightgbm || exit -1

cd $BUILD_SOURCESDIRECTORY/python-package && python setup.py install --precompile || exit -1
pytest $BUILD_SOURCESDIRECTORY/tests || exit -1

if [[ $TASK == "regular" ]]; then
    if [[ $AGENT_OS == "Darwin" ]]; then
        cp ${BUILD_REPOSITORY_LOCALPATH}/lib_lightgbm.so ${BUILD_ARTIFACTSTAGINGDIRECTORY}/lib_lightgbm.dylib
    else
        cp $BUILD_SOURCESDIRECTORY/lib_lightgbm.so $BUILD_ARTIFACTSTAGINGDIRECTORY/lib_lightgbm.so
    fi
    cd $BUILD_SOURCESDIRECTORY/examples/python-guide
    sed -i'.bak' '/import lightgbm as lgb/a\
import matplotlib\
matplotlib.use\(\"Agg\"\)\
' plot_example.py  # prevent interactive window mode
    sed -i'.bak' 's/graph.render(view=True)/graph.render(view=False)/' plot_example.py
    for f in *.py; do python $f || exit -1; done  # run all examples
fi
