#!/bin/bash

if [[ $AGENT_OS == "Darwin" ]]; then
    if  [[ $COMPILER == "clang" ]]; then
        brew install libomp
        brew reinstall cmake  # CMake >=3.12 is needed to find OpenMP at macOS
    else
        if [[ $TASK != "mpi" ]]; then
            brew install gcc
        fi
    fi
    if [[ $TASK == "mpi" ]]; then
        brew install open-mpi
    fi
else  # Linux
    if [[ $COMPILER == "clang" ]]; then
        update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-6.0 100
        update-alternatives --install /usr/bin/clang clang /usr/bin/clang-6.0 100
        sudo apt-get update
        sudo apt-get install libomp-dev
    fi
    if [[ $TASK == "mpi" ]]; then
        sudo apt-get update
        sudo apt-get install -y libopenmpi-dev openmpi-bin
    fi
    if [[ $TASK == "gpu" ]]; then
        sudo apt-get update
        sudo apt-get install --no-install-recommends -y ocl-icd-opencl-dev libboost-dev libboost-system-dev libboost-filesystem-dev
        cd $AGENT_HOMEDIRECTORY
        wget -q https://github.com/Microsoft/LightGBM/releases/download/v2.0.12/AMD-APP-SDKInstaller-v3.0.130.136-GA-linux64.tar.bz2
        tar -xjf AMD-APP-SDK*.tar.bz2
        mkdir -p $OPENCL_VENDOR_PATH
        mkdir -p $AMDAPPSDK_PATH
        sh AMD-APP-SDK*.sh --tar -xf -C $AMDAPPSDK_PATH
        mv $AMDAPPSDK_PATH/lib/x86_64/sdk/* $AMDAPPSDK_PATH/lib/x86_64/
        echo libamdocl64.so > $OPENCL_VENDOR_PATH/amdocl64.icd
    fi
fi
