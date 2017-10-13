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

case ${TRAVIS_OS_NAME} in
    osx)
        export CXX=g++-7
        export CC=gcc-7
        ;;
    linux)
        ;;
esac

cd $TRAVIS_BUILD_DIR

if [[ ${TASK} == "check-docs" ]]; then
    cd docs
    sudo apt-get install linkchecker
    pip install rstcheck sphinx sphinx_rtd_theme  # html5validator
    rstcheck --report warning --ignore-directives=autoclass,autofunction `find . -type f -name "*.rst"` || exit -1
    make html || exit -1
    find ./_build/html/ -type f -name '*.html' -exec \
    sed -i -e 's;\(\.\/[^.]*\.\)rst\([^[:space:]]*\);\1html\2;g' {} \;  # Emulate js function
#    html5validator --root ./_build/html/ || exit -1
    linkchecker --config=.linkcheckerrc ./_build/html/*.html || exit -1
    exit 0
fi

if [[ ${TASK} == "pylint" ]]; then
    pip install pep8
    pep8 --ignore=E501 --exclude=./compute,./docs . || exit -1
    exit 0
fi

if [[ ${TASK} == "if-else" ]]; then
    conda create -q -n test-env python=$PYTHON_VERSION numpy
    source activate test-env
    mkdir build && cd build && cmake .. && make lightgbm || exit -1
    cd $TRAVIS_BUILD_DIR/tests/cpp_test && ../../lightgbm config=train.conf convert_model_language=cpp convert_model=../../src/boosting/gbdt_prediction.cpp && ../../lightgbm config=predict.conf output_result=origin.pred || exit -1
    cd $TRAVIS_BUILD_DIR/build && make lightgbm || exit -1
    cd $TRAVIS_BUILD_DIR/tests/cpp_test && ../../lightgbm config=predict.conf output_result=ifelse.pred && python test.py || exit -1
    exit 0
fi

if [[ ${TASK} == "proto" ]]; then
    conda create -q -n test-env python=$PYTHON_VERSION numpy
    source activate test-env
    mkdir build && cd build && cmake .. && make lightgbm || exit -1
    cd $TRAVIS_BUILD_DIR/tests/cpp_test && ../../lightgbm config=train.conf && ../../lightgbm config=predict.conf output_result=origin.pred || exit -1
    cd $TRAVIS_BUILD_DIR && git clone https://github.com/google/protobuf && cd protobuf && ./autogen.sh && ./configure && make && sudo make install && sudo ldconfig
    cd $TRAVIS_BUILD_DIR/build && rm -rf * && cmake -DUSE_PROTO=ON .. && make lightgbm || exit -1
    cd $TRAVIS_BUILD_DIR/tests/cpp_test && ../../lightgbm config=train.conf model_format=proto && ../../lightgbm config=predict.conf output_result=proto.pred model_format=proto || exit -1
    cd $TRAVIS_BUILD_DIR/tests/cpp_test && python test.py || exit -1
    exit 0
fi

conda create -q -n test-env python=$PYTHON_VERSION numpy nose scipy scikit-learn pandas matplotlib pytest
source activate test-env

if [[ ${TASK} == "sdist" ]]; then
    LGB_VER=$(head -n 1 VERSION.txt)
    cd $TRAVIS_BUILD_DIR/python-package && python setup.py sdist || exit -1
    cd $TRAVIS_BUILD_DIR/python-package/dist && pip install lightgbm-$LGB_VER.tar.gz -v || exit -1
    cd $TRAVIS_BUILD_DIR && pytest tests/python_package_test || exit -1
    exit 0
elif [[ ${TASK} == "bdist" ]]; then
    LGB_VER=$(head -n 1 VERSION.txt)
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
        export PATH="$AMDAPPSDK/include/:$PATH"
        export BOOST_ROOT="$HOME/miniconda/envs/test-env/"
        LGB_VER=$(head -n 1 VERSION.txt)
        sed -i 's/const std::string kDefaultDevice = "cpu";/const std::string kDefaultDevice = "gpu";/' ../include/LightGBM/config.h
        cd $TRAVIS_BUILD_DIR/python-package && python setup.py sdist || exit -1
        cd $TRAVIS_BUILD_DIR/python-package/dist && pip install lightgbm-$LGB_VER.tar.gz -v --install-option=--gpu || exit -1
        cd $TRAVIS_BUILD_DIR && pytest tests/python_package_test || exit -1
        exit 0
    fi
fi

mkdir build && cd build

if [[ ${TASK} == "mpi" ]]; then
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
