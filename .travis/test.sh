#!/bin/bash

if [[ $TRAVIS_OS_NAME == "osx" ]] && [[ $COMPILER == "gcc" ]]; then
    export CXX=g++-8
    export CC=gcc-8
elif [[ $TRAVIS_OS_NAME == "linux" ]] && [[ $COMPILER == "clang" ]]; then
    export CXX=clang++
    export CC=clang
fi

conda create -q -n test-env python=$PYTHON_VERSION
source activate test-env

cd $TRAVIS_BUILD_DIR

if [[ $TASK == "check-docs" ]]; then
    if [[ $PYTHON_VERSION == "2.7" ]]; then
        conda install mock
    fi
    conda install sphinx "sphinx_rtd_theme>=0.3"  # html5validator
    pip install rstcheck
    # check reStructuredText formatting
    cd $TRAVIS_BUILD_DIR/python-package
    rstcheck --report warning `find . -type f -name "*.rst"` || exit -1
    cd $TRAVIS_BUILD_DIR/docs
    rstcheck --report warning --ignore-directives=autoclass,autofunction `find . -type f -name "*.rst"` || exit -1
    # build docs and check them for broken links
    make html || exit -1
    find ./_build/html/ -type f -name '*.html' -exec \
    sed -i'.bak' -e 's;\(\.\/[^.]*\.\)rst\([^[:space:]]*\);\1html\2;g' {} \;  # emulate js function
#    html5validator --root ./_build/html/ || exit -1
    if [[ $TRAVIS_OS_NAME == "linux" ]]; then
        sudo apt-get install linkchecker
        linkchecker --config=.linkcheckerrc ./_build/html/*.html || exit -1
    fi
    # check the consistency of parameters' descriptions and other stuff
    cp $TRAVIS_BUILD_DIR/docs/Parameters.rst $TRAVIS_BUILD_DIR/docs/Parameters-backup.rst
    cp $TRAVIS_BUILD_DIR/src/io/config_auto.cpp $TRAVIS_BUILD_DIR/src/io/config_auto-backup.cpp
    python $TRAVIS_BUILD_DIR/helper/parameter_generator.py || exit -1
    diff $TRAVIS_BUILD_DIR/docs/Parameters-backup.rst $TRAVIS_BUILD_DIR/docs/Parameters.rst || exit -1
    diff $TRAVIS_BUILD_DIR/src/io/config_auto-backup.cpp $TRAVIS_BUILD_DIR/src/io/config_auto.cpp || exit -1
    exit 0
fi

if [[ $TASK == "pylint" ]]; then
    conda install pycodestyle
    pycodestyle --ignore=E501,W503 --exclude=./compute,./docs,./.nuget . || exit -1
    exit 0
fi

if [[ $TASK == "if-else" ]]; then
    conda install numpy
    mkdir $TRAVIS_BUILD_DIR/build && cd $TRAVIS_BUILD_DIR/build && cmake .. && make lightgbm || exit -1
    cd $TRAVIS_BUILD_DIR/tests/cpp_test && ../../lightgbm config=train.conf convert_model_language=cpp convert_model=../../src/boosting/gbdt_prediction.cpp && ../../lightgbm config=predict.conf output_result=origin.pred || exit -1
    cd $TRAVIS_BUILD_DIR/build && make lightgbm || exit -1
    cd $TRAVIS_BUILD_DIR/tests/cpp_test && ../../lightgbm config=predict.conf output_result=ifelse.pred && python test.py || exit -1
    exit 0
fi

conda install numpy nose scipy scikit-learn pandas matplotlib python-graphviz pytest

if [[ $TRAVIS_OS_NAME == "osx" ]] && [[ $COMPILER == "clang" ]]; then
    ln -sf `ls -d "$(brew --cellar libomp)"/*/lib`/* $CONDA_PREFIX/lib || exit -1  # fix "OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized." (OpenMP library conflict due to conda's MKL)
fi

if [[ $TASK == "sdist" ]]; then
    cd $TRAVIS_BUILD_DIR/python-package && python setup.py sdist || exit -1
    pip install $TRAVIS_BUILD_DIR/python-package/dist/lightgbm-$LGB_VER.tar.gz -v || exit -1
    pytest $TRAVIS_BUILD_DIR/tests/python_package_test || exit -1
    exit 0
elif [[ $TASK == "bdist" ]]; then
    if [[ $TRAVIS_OS_NAME == "osx" ]]; then
        cd $TRAVIS_BUILD_DIR/python-package && python setup.py bdist_wheel --plat-name=macosx --universal || exit -1
        mv dist/lightgbm-$LGB_VER-py2.py3-none-macosx.whl dist/lightgbm-$LGB_VER-py2.py3-none-macosx_10_9_x86_64.macosx_10_10_x86_64.macosx_10_11_x86_64.macosx_10_12_x86_64.macosx_10_13_x86_64.whl
    else
        cd $TRAVIS_BUILD_DIR/python-package && python setup.py bdist_wheel --plat-name=manylinux1_x86_64 --universal || exit -1
    fi
    pip install $TRAVIS_BUILD_DIR/python-package/dist/*.whl || exit -1
    pytest $TRAVIS_BUILD_DIR/tests || exit -1
    exit 0
fi

if [[ $TASK == "gpu" ]]; then
    conda install --yes -c conda-forge boost
    sed -i'.bak' 's/std::string device_type = "cpu";/std::string device_type = "gpu";/' $TRAVIS_BUILD_DIR/include/LightGBM/config.h
    grep -q 'std::string device_type = "gpu"' $TRAVIS_BUILD_DIR/include/LightGBM/config.h || exit -1  # make sure that changes were really done
    if [[ $METHOD == "pip" ]]; then
        cd $TRAVIS_BUILD_DIR/python-package && python setup.py sdist || exit -1
        pip install $TRAVIS_BUILD_DIR/python-package/dist/lightgbm-$LGB_VER.tar.gz -v --install-option=--gpu --install-option="--boost-root=$HOME/miniconda/envs/test-env/" --install-option="--opencl-include-dir=$AMDAPPSDK_PATH/include/" || exit -1
        pytest $TRAVIS_BUILD_DIR/tests/python_package_test || exit -1
        exit 0
    fi
fi

mkdir $TRAVIS_BUILD_DIR/build && cd $TRAVIS_BUILD_DIR/build

if [[ $TASK == "mpi" ]]; then
    if [[ $METHOD == "pip" ]]; then
        cd $TRAVIS_BUILD_DIR/python-package && python setup.py sdist || exit -1
        pip install $TRAVIS_BUILD_DIR/python-package/dist/lightgbm-$LGB_VER.tar.gz -v --install-option=--mpi || exit -1
        pytest $TRAVIS_BUILD_DIR/tests/python_package_test || exit -1
        exit 0
    fi
    cmake -DUSE_MPI=ON ..
elif [[ $TASK == "gpu" ]]; then
    cmake -DUSE_GPU=ON -DBOOST_ROOT=$HOME/miniconda/envs/test-env/ -DOpenCL_INCLUDE_DIR=$AMDAPPSDK_PATH/include/ ..
else
    cmake ..
fi

make _lightgbm || exit -1

cd $TRAVIS_BUILD_DIR/python-package && python setup.py install --precompile || exit -1
pytest $TRAVIS_BUILD_DIR/tests || exit -1

if [[ $TASK == "regular" ]]; then
    cd $TRAVIS_BUILD_DIR/examples/python-guide
    sed -i'.bak' '/import lightgbm as lgb/a\
import matplotlib\
matplotlib.use\(\"Agg\"\)\
' plot_example.py  # prevent interactive window mode
    sed -i'.bak' 's/graph.render(view=True)/graph.render(view=False)/' plot_example.py
    for f in *.py; do python $f || exit -1; done  # run all examples
fi
