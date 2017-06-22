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
    export LIBRARY_PATH="$HOME/miniconda/lib:$LIBRARY_PATH"
    export LD_RUN_PATH="$HOME/miniconda/lib:$LD_RUN_PATH"
    export CPLUS_INCLUDE_PATH="$HOME/miniconda/include:$AMDAPPSDK/include/:$CPLUS_INCLUDE_PATH"
fi

cd $TRAVIS_BUILD_DIR

if [[ ${TASK} == "pylint" ]]; then
    pip install pep8
    pep8 --ignore=E501 --exclude=./compute,./docs . || exit -1
    exit 0
fi

conda install --yes numpy scipy scikit-learn pandas matplotlib
pip install pytest

if [[ $TRAVIS_OS_NAME == "linux" ]]; then 
    conda install --yes atlas
    conda install --yes -c conda-forge boost=1.63.0
fi

if [[ ${TASK} == "pip" ]]; then
    LGB_VER=$(head -n 1 VERSION.txt)
    cd $TRAVIS_BUILD_DIR/python-package && python setup.py sdist || exit -1
    cd $TRAVIS_BUILD_DIR/python-package/dist && pip install lightgbm-$LGB_VER.tar.gz -v || exit -1
    cd $TRAVIS_BUILD_DIR && pytest tests/python_package_test || exit -1
    exit 0
fi

mkdir build && cd build

if [[ ${TASK} == "regular" ]]; then
    cmake ..
elif [[ ${TASK} == "mpi" ]]; then
    cmake -DUSE_MPI=ON ..
elif [[ ${TASK} == "gpu" ]]; then
    cmake -DUSE_GPU=ON -DBOOST_ROOT="$HOME/miniconda/" -DOpenCL_INCLUDE_DIR=$AMDAPPSDK/include/ ..
    sed -i 's/std::string device_type = "cpu";/std::string device_type = "gpu";/' ../include/LightGBM/config.h
fi

make _lightgbm || exit -1

cd $TRAVIS_BUILD_DIR/python-package && python setup.py install --precompile || exit -1
cd $TRAVIS_BUILD_DIR && pytest . || exit -1
