#!/bin/bash

if [[ $OS_NAME == "macos" ]]; then
    if  [[ $COMPILER == "clang" ]]; then
        brew install libomp
        if [[ $AZURE == "true" ]]; then
            sudo xcode-select -s /Applications/Xcode_9.4.1.app/Contents/Developer
        fi
    else  # gcc
        if [[ $TASK != "mpi" ]]; then
            brew install gcc
        fi
    fi
    if [[ $TASK == "mpi" ]]; then
        brew install open-mpi
    fi
    if [[ $AZURE == "true" ]] && [[ $TASK == "sdist" ]]; then
        brew install swig@3
    fi
    wget -q -O conda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
else  # Linux
    if [[ $TASK == "mpi" ]]; then
        sudo apt-get update
        sudo apt-get install --no-install-recommends -y libopenmpi-dev openmpi-bin
    fi
    if [[ $TASK == "gpu" ]]; then
        sudo add-apt-repository ppa:mhier/libboost-latest -y
        sudo apt-get update
        sudo apt-get install --no-install-recommends -y libboost1.74-dev ocl-icd-opencl-dev
        cd $BUILD_DIRECTORY  # to avoid permission errors
        wget -q https://github.com/microsoft/LightGBM/releases/download/v2.0.12/AMD-APP-SDKInstaller-v3.0.130.136-GA-linux64.tar.bz2
        tar -xjf AMD-APP-SDK*.tar.bz2
        mkdir -p $OPENCL_VENDOR_PATH
        mkdir -p $AMDAPPSDK_PATH
        sh AMD-APP-SDK*.sh --tar -xf -C $AMDAPPSDK_PATH
        mv $AMDAPPSDK_PATH/lib/x86_64/sdk/* $AMDAPPSDK_PATH/lib/x86_64/
        echo libamdocl64.so > $OPENCL_VENDOR_PATH/amdocl64.icd
    fi
    if [[ $TASK == "cuda" ]]; then
        apt-get update
        apt-get install --no-install-recommends -y curl wget
        curl -sL https://cmake.org/files/v3.18/cmake-3.18.1-Linux-x86_64.sh -o cmake.sh
        chmod +x cmake.sh
        ./cmake.sh --prefix=/usr/local --exclude-subdir
    fi
    if [[ $TRAVIS == "true" ]] || [[ $GITHUB_ACTIONS == "true" ]]; then
        wget -q -O conda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    fi
fi

if [[ "${TASK:0:9}" != "r-package" ]]; then
    if [[ $TRAVIS == "true" ]] || [[ $OS_NAME == "macos" ]]; then
        sh conda.sh -b -p $CONDA
    fi
    conda config --set always_yes yes --set changeps1 no
    conda update -q -y conda
fi
