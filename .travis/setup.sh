#!/bin/bash

if [[ $TRAVIS_OS_NAME == "osx" ]]; then
    brew install cmake
    brew install openmpi
    brew install gcc --without-multilib
    wget -O conda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
else
    if [[ ${TASK} != "pylint" ]]; then
        sudo add-apt-repository ppa:george-edison55/cmake-3.x -y
        sudo apt-get update -q
        sudo apt-get install -y cmake
        sudo apt-get install -y libopenmpi-dev openmpi-bin build-essential
    fi
    wget -O conda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
fi

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
