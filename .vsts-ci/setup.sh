#!/bin/bash
test -n $CC  && unset CC
test -n $CXX && unset CXX
export LGB_VER=$(head -n 1 VERSION.txt)
export AMDAPPSDK=$HOME/AMDAPPSDK
export LD_LIBRARY_PATH="AMDAPPSDK/lib/x86_64:LD_LIBRARY_PATH"
export OPENCL_VENDOR_PATH=$AMDAPPSDK/etc/OpenCL/vendors
if [[ ${AGENT_OS} == "darwin" ]]; then
    rm '/usr/local/include/c++'
#    brew cask uninstall oclint  #  reserve variant to deal with conflict link
    if [[ $TASK == "mpi" ]]; then
        brew install open-mpi
    else
        brew install gcc
    fi
#    brew link --overwrite gcc  # previous variant to deal with conflict link
else
    if [[ $TASK == "mpi" ]]; then
        sudo apt-get install -y libopenmpi-dev openmpi-bin
    fi
    if [[ $TASK == "gpu" ]]; then
        sudo apt-get install -y ocl-icd-opencl-dev
    fi
fi


if [[ $TASK == "gpu" ]] && [[ ${AGENT_OS} == "linux" ]]; then
    wget https://github.com/Microsoft/LightGBM/releases/download/v2.0.12/AMD-APP-SDKInstaller-v3.0.130.136-GA-linux64.tar.bz2
    tar -xjf AMD-APP-SDK*.tar.bz2
    mkdir -p $OPENCL_VENDOR_PATH
    sh AMD-APP-SDK*.sh --tar -xf -C $AMDAPPSDK
    mv $AMDAPPSDK/lib/x86_64/sdk/* $AMDAPPSDK/lib/x86_64/
    echo libamdocl64.so > $OPENCL_VENDOR_PATH/amdocl64.icd
fi
