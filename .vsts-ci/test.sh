#!/bin/bash
if [[ $AGENT_OS == "darwin" ]]; then
    export CXX=g++-8
    export CC=gcc-8
fi


cd ${BUILD_REPOSITORY_LOCALPATH}

if [[ $TASK == "check-docs" ]]; then
    if [[ $AGENT_OS == "linux" ]]; then
        sudo apt-get install linkchecker -y
    fi
    if [[ ${PYTHON_VERSION} == "2.7" ]]; then
        conda install -y -n $CONDA_ENV mock
    fi
    conda install -y -n $CONDA_ENV sphinx "sphinx_rtd_theme>=0.3"  # html5validator
    pip install rstcheck
    cd ${BUILD_REPOSITORY_LOCALPATH}/python-package
    rstcheck --report warning `find . -type f -name "*.rst"` || exit -1
    cd ${BUILD_REPOSITORY_LOCALPATH}/docs
    rstcheck --report warning --ignore-directives=autoclass,autofunction `find . -type f -name "*.rst"` || exit -1
    make html || exit -1
    find ./_build/html/ -type f -name '*.html' -exec \
    sed -i -e 's;\(\.\/[^.]*\.\)rst\([^[:space:]]*\);\1html\2;g' {} \;  # emulate js function
#    html5validator --root ./_build/html/ || exit -1
    if [[ $AGENT_OS == "linux" ]]; then
        linkchecker --config=.linkcheckerrc ./_build/html/*.html || exit -1
    fi
    exit 0
fi

if [[ $TASK == "pylint" ]]; then
    conda install -y -n $CONDA_ENV pycodestyle
    pycodestyle --ignore=E501,W503 --exclude=./compute,./docs,./.nuget . || exit -1
    exit 0
fi

if [[ $TASK == "if-else" ]]; then
    conda install -y -n $CONDA_ENV numpy
    mkdir build && cd build && cmake .. && make lightgbm || exit -1
    cd ${BUILD_REPOSITORY_LOCALPATH}/tests/cpp_test && ../../lightgbm config=train.conf convert_model_language=cpp convert_model=../../src/boosting/gbdt_prediction.cpp && ../../lightgbm config=predict.conf output_result=origin.pred || exit -1
    cd ${BUILD_REPOSITORY_LOCALPATH}/build && make lightgbm || exit -1
    cd ${BUILD_REPOSITORY_LOCALPATH}/tests/cpp_test && ../../lightgbm config=predict.conf output_result=ifelse.pred && python test.py || exit -1
    exit 0
fi

conda install -y -n $CONDA_ENV numpy nose scipy scikit-learn pandas matplotlib python-graphviz pytest

if [[ $TASK == "sdist" ]]; then
    cd ${BUILD_REPOSITORY_LOCALPATH}/python-package && python setup.py sdist || exit -1
    pip install ${BUILD_REPOSITORY_LOCALPATH}/python-package/dist/lightgbm-$LGB_VER.tar.gz -v || exit -1
    pytest ${BUILD_REPOSITORY_LOCALPATH}/tests/python_package_test || exit -1
    exit 0
elif [[ $TASK == "bdist" ]]; then
    if [[ $AGENT_OS == "darwin" ]]; then
        cd ${BUILD_REPOSITORY_LOCALPATH}/python-package && python setup.py bdist_wheel --plat-name=macdarwin --universal || exit -1
        mv dist/lightgbm-$LGB_VER-py2.py3-none-macdarwin.whl dist/lightgbm-$LGB_VER-py2.py3-none-macdarwin_10_9_x86_64.macdarwin_10_10_x86_64.macdarwin_10_11_x86_64.macdarwin_10_12_x86_64.macdarwin_10_13_x86_64.whl
    else
        cd ${BUILD_REPOSITORY_LOCALPATH}/python-package && python setup.py bdist_wheel --plat-name=manylinux1_x86_64 --universal || exit -1
    fi
    pip install ${BUILD_REPOSITORY_LOCALPATH}/python-package/dist/*.whl || exit -1
    pytest ${BUILD_REPOSITORY_LOCALPATH}/tests/python_package_test || exit -1
    exit 0
fi

if [[ $TASK == "gpu" ]]; then
    conda install --yes -c conda-forge boost
    sed -i 's/std::string device_type = "cpu";/std::string device_type = "gpu";/' ${BUILD_REPOSITORY_LOCALPATH}/include/LightGBM/config.h
    grep -q 'std::string device_type = "gpu"' ${BUILD_REPOSITORY_LOCALPATH}/include/LightGBM/config.h || exit -1  # make sure that changes were really done
    if [[ $METHOD == "pip" ]]; then
        cd ${BUILD_REPOSITORY_LOCALPATH}/python-package && python setup.py sdist || exit -1
        pip install ${BUILD_REPOSITORY_LOCALPATH}/python-package/dist/lightgbm-$LGB_VER.tar.gz -v --install-option=--gpu --install-option="--boost-root=/usr/share/miniconda/envs/$CONDA_ENV/" --install-option="--opencl-include-dir=$AMDAPPSDK/include/" || exit -1
        pytest ${BUILD_REPOSITORY_LOCALPATH}/tests/python_package_test || exit -1
        exit 0
    fi
fi

mkdir build && cd build

if [[ $TASK == "mpi" ]]; then
    cd ${BUILD_REPOSITORY_LOCALPATH}/python-package && python setup.py sdist || exit -1
    pip install ${BUILD_REPOSITORY_LOCALPATH}/python-package/dist/lightgbm-$LGB_VER.tar.gz -v --install-option=--mpi || exit -1
    cd ${BUILD_REPOSITORY_LOCALPATH}/build
    cmake -DUSE_MPI=ON ..
elif [[ $TASK == "gpu" ]]; then
    cmake -DUSE_GPU=ON -DBOOST_ROOT=/usr/share/miniconda/envs/$CONDA_ENV/ -DOpenCL_INCLUDE_DIR=$AMDAPPSDK/include/ ..
else
    cmake ..
fi

make _lightgbm || exit -1

cd ${BUILD_REPOSITORY_LOCALPATH}/python-package && python setup.py install --precompile || exit -1
pytest ${BUILD_REPOSITORY_LOCALPATH} || exit -1

if [[ $TASK == "regular" ]]; then
    cd ${BUILD_REPOSITORY_LOCALPATH}/examples/python-guide
    sed -i'.bak' '/import lightgbm as lgb/a\
import matplotlib\
matplotlib.use\(\"Agg\"\)\
' plot_example.py  # prevent interactive window mode
    sed -i 's/graph.render(view=True)/graph.render(view=False)/' plot_example.py
    for f in *.py; do python $f || exit -1; done  # run all examples
fi
