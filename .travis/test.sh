if [[ ${TASK} == "gpu" ]]; then
    bash .travis/amd_sdk.sh;
    tar -xjf AMD-SDK.tar.bz2;
    AMDAPPSDK=${HOME}/AMDAPPSDK;
    export OPENCL_VENDOR_PATH=${AMDAPPSDK}/etc/OpenCL/vendors;
    mkdir -p ${OPENCL_VENDOR_PATH};
    sh AMD-APP-SDK*.sh --tar -xf -C ${AMDAPPSDK};
    echo libamdocl64.so > ${OPENCL_VENDOR_PATH}/amdocl64.icd;
    export LD_LIBRARY_PATH=${AMDAPPSDK}/lib/x86_64:${LD_LIBRARY_PATH};
    chmod +x ${AMDAPPSDK}/bin/x86_64/clinfo;
    ${AMDAPPSDK}/bin/x86_64/clinfo;
    export LIBRARY_PATH="$HOME/miniconda/envs/test-env/lib:$LIBRARY_PATH"
    export LD_RUN_PATH="$HOME/miniconda/envs/test-env/lib:$LD_RUN_PATH"
    export CPLUS_INCLUDE_PATH="$HOME/miniconda/envs/test-env/include:$AMDAPPSDK/include/:$CPLUS_INCLUDE_PATH"
fi

if [[ $TRAVIS_OS_NAME == "osx" ]]; then
    export CXX=g++-8
    export CC=gcc-8
fi

LGB_VER=$(head -n 1 VERSION.txt)

conda create -q -n test-env python=$PYTHON_VERSION
source activate test-env

cd $TRAVIS_BUILD_DIR

if [[ ${TASK} == "check-docs" ]]; then
    if [[ $TRAVIS_OS_NAME != "osx" ]]; then
        sudo apt-get install linkchecker
    fi
    if [[ ${PYTHON_VERSION} == "2.7" ]]; then
        conda install mock
    fi
    conda install sphinx sphinx_rtd_theme  # html5validator
    pip install rstcheck
    cd python-package
    rstcheck --report warning `find . -type f -name "*.rst"` || exit -1
    cd ../docs
    rstcheck --report warning --ignore-directives=autoclass,autofunction `find . -type f -name "*.rst"` || exit -1
    make html || exit -1
    find ./_build/html/ -type f -name '*.html' -exec \
    sed -i -e 's;\(\.\/[^.]*\.\)rst\([^[:space:]]*\);\1html\2;g' {} \;  # Emulate js function
#    html5validator --root ./_build/html/ || exit -1
    if [[ $TRAVIS_OS_NAME != "osx" ]]; then
        linkchecker --config=.linkcheckerrc ./_build/html/*.html || exit -1
    fi
    exit 0
fi

if [[ ${TASK} == "pylint" ]]; then
    conda install pycodestyle
    pycodestyle --ignore=E501,W503 --exclude=./compute,./docs,./.nuget . || exit -1
    exit 0
fi

if [[ ${TASK} == "if-else" ]]; then
    conda install numpy
    mkdir build && cd build && cmake .. && make lightgbm || exit -1
    cd $TRAVIS_BUILD_DIR/tests/cpp_test && ../../lightgbm config=train.conf convert_model_language=cpp convert_model=../../src/boosting/gbdt_prediction.cpp && ../../lightgbm config=predict.conf output_result=origin.pred || exit -1
    cd $TRAVIS_BUILD_DIR/build && make lightgbm || exit -1
    cd $TRAVIS_BUILD_DIR/tests/cpp_test && ../../lightgbm config=predict.conf output_result=ifelse.pred && python test.py || exit -1
    exit 0
fi

conda install numpy nose scipy scikit-learn pandas matplotlib pytest

if [[ ${TASK} == "sdist" ]]; then
    cd $TRAVIS_BUILD_DIR/python-package && python setup.py sdist || exit -1
    cd $TRAVIS_BUILD_DIR/python-package/dist && pip install lightgbm-$LGB_VER.tar.gz -v || exit -1
    cd $TRAVIS_BUILD_DIR && pytest tests/python_package_test || exit -1
    exit 0
elif [[ ${TASK} == "bdist" ]]; then
    if [[ $TRAVIS_OS_NAME == "osx" ]]; then
        cd $TRAVIS_BUILD_DIR/python-package && python setup.py bdist_wheel --plat-name=macosx --universal || exit -1
        mv dist/lightgbm-${LGB_VER}-py2.py3-none-macosx.whl dist/lightgbm-${LGB_VER}-py2.py3-none-macosx_10_9_x86_64.macosx_10_10_x86_64.macosx_10_11_x86_64.macosx_10_12_x86_64.whl
    else
        cd $TRAVIS_BUILD_DIR/python-package && python setup.py bdist_wheel --plat-name=manylinux1_x86_64 --universal || exit -1
    fi
    cd $TRAVIS_BUILD_DIR/python-package && pip install dist/*.whl || exit -1
    cd $TRAVIS_BUILD_DIR && pytest tests/python_package_test || exit -1
    exit 0
fi

if [[ ${TASK} == "gpu" ]]; then 
    conda install --yes -c conda-forge boost=1.63.0
    if [[ ${METHOD} == "pip" ]]; then
        sed -i 's/const std::string kDefaultDevice = "cpu";/const std::string kDefaultDevice = "gpu";/' ../include/LightGBM/config.h
        cd $TRAVIS_BUILD_DIR/python-package && python setup.py sdist || exit -1
        cd $TRAVIS_BUILD_DIR/python-package/dist && pip install lightgbm-$LGB_VER.tar.gz -v --install-option=--gpu --install-option="--boost-root=$HOME/miniconda/envs/test-env/" --install-option="--opencl-include-dir=$AMDAPPSDK/include/" || exit -1
        cd $TRAVIS_BUILD_DIR && pytest tests/python_package_test || exit -1
        exit 0
    fi
fi

mkdir build && cd build

if [[ ${TASK} == "mpi" ]]; then
    cd $TRAVIS_BUILD_DIR/python-package && python setup.py sdist || exit -1
    cd $TRAVIS_BUILD_DIR/python-package/dist && pip install lightgbm-$LGB_VER.tar.gz -v --install-option=--mpi || exit -1
    cd $TRAVIS_BUILD_DIR/build
    cmake -DUSE_MPI=ON ..
elif [[ ${TASK} == "gpu" ]]; then
    cmake -DUSE_GPU=ON -DBOOST_ROOT="$HOME/miniconda/envs/test-env/" -DOpenCL_INCLUDE_DIR=$AMDAPPSDK/include/ ..
    sed -i 's/const std::string kDefaultDevice = "cpu";/const std::string kDefaultDevice = "gpu";/' ../include/LightGBM/config.h
else
    cmake ..
fi

make _lightgbm || exit -1

cd $TRAVIS_BUILD_DIR/python-package && python setup.py install --precompile || exit -1
cd $TRAVIS_BUILD_DIR && pytest . || exit -1

if [[ ${TASK} == "regular" ]]; then
    cd $TRAVIS_BUILD_DIR/examples/python-guide && python simple_example.py && python sklearn_example.py && python advanced_example.py || exit -1
fi
