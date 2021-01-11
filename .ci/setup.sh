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
    if [[ $IN_UBUNTU_LATEST_CONTAINER == "true" ]]; then
        sudo apt-get update || exit -1
        sudo apt-get install -y --no-install-recommends \
            software-properties-common || exit  -1

        sudo add-apt-repository -y ppa:git-core/ppa
        sudo apt-get update

        sudo apt-get install -y --no-install-recommends \
            apt-utils \
            build-essential \
            ca-certificates \
            curl \
            git \
            iputils-ping \
            jq \
            libcurl4 \
            libicu66 \
            libssl1.1 \
            libunwind8 \
            locales \
            netcat \
            sudo \
            unzip \
            wget \
            zip

        export LANG="en_US.UTF-8"
        sudo locale-gen ${LANG}
        update-locale

        sudo apt-get install -y --no-install-recommends \
            cmake \
            clang-11
    fi
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
    if [[ $SETUP_CONDA != "false" ]]; then
        wget -q -O conda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    fi
fi

if [[ "${TASK}" != "r-package" ]]; then
    if [[ $SETUP_CONDA != "false" ]]; then
        sh conda.sh -b -p $CONDA
    fi
    conda config --set always_yes yes --set changeps1 no
    conda update -q -y conda
fi
