#!/bin/bash

if [[ $OS_NAME == "macos" ]]; then
    if  [[ $COMPILER == "clang" ]]; then
        brew install libomp
        if [[ $AZURE == "true" ]]; then
            sudo xcode-select -s /Applications/Xcode_9.4.1.app/Contents/Developer || exit -1
        fi
    else  # gcc
        if [[ $TASK != "mpi" ]]; then
            brew install gcc
        fi
    fi
    if [[ $TASK == "mpi" ]]; then
        brew install open-mpi
    fi
    if [[ $TASK == "swig" ]]; then
        brew install swig
    fi
    brew install graphviz
    curl -sL -o conda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
else  # Linux
    if [[ $IN_UBUNTU_LATEST_CONTAINER == "true" ]]; then
        # fixes error "unable to initialize frontend: Dialog"
        # https://github.com/moby/moby/issues/27988#issuecomment-462809153
        echo 'debconf debconf/frontend select Noninteractive' | sudo debconf-set-selections

        sudo apt-get update
        sudo apt-get install --no-install-recommends -y \
            software-properties-common

        sudo apt-get install --no-install-recommends -y \
            apt-utils \
            build-essential \
            ca-certificates \
            cmake \
            curl \
            git \
            iputils-ping \
            jq \
            libcurl4 \
            libicu66 \
            libssl1.1 \
            libunwind8 \
            libxau6 \
            libxext6 \
            libxrender1 \
            locales \
            netcat \
            unzip \
            zip
        if [[ $COMPILER == "clang" ]]; then
            sudo apt-get install --no-install-recommends -y \
                clang \
                libomp-dev
        fi

        export LANG="en_US.UTF-8"
        export LC_ALL="${LANG}"
        sudo locale-gen ${LANG}
        sudo update-locale
    fi
    if [[ $TASK == "mpi" ]]; then
        sudo apt-get update
        sudo apt-get install --no-install-recommends -y \
            libopenmpi-dev \
            openmpi-bin
    fi
    if [[ $TASK == "gpu" ]]; then
        sudo add-apt-repository ppa:mhier/libboost-latest -y
        sudo apt-get update
        sudo apt-get install --no-install-recommends -y \
            libboost1.74-dev \
            ocl-icd-opencl-dev
        cd $BUILD_DIRECTORY  # to avoid permission errors
        curl -sL -o AMD-APP-SDKInstaller.tar.bz2 https://github.com/microsoft/LightGBM/releases/download/v2.0.12/AMD-APP-SDKInstaller-v3.0.130.136-GA-linux64.tar.bz2
        tar -xjf AMD-APP-SDKInstaller.tar.bz2
        mkdir -p $OPENCL_VENDOR_PATH
        mkdir -p $AMDAPPSDK_PATH
        sh AMD-APP-SDK*.sh --tar -xf -C $AMDAPPSDK_PATH
        mv $AMDAPPSDK_PATH/lib/x86_64/sdk/* $AMDAPPSDK_PATH/lib/x86_64/
        echo libamdocl64.so > $OPENCL_VENDOR_PATH/amdocl64.icd
    fi
    if [[ $TASK == "transform" ]]; then
        sudo apt-get update && apt-get -y upgrade
        sudo DEBIAN_FRONTEND="noninteractive" apt-get install -y libboost-all-dev gcc g++ wget cmake git curl
        cmake --version
        gcc --version
        cd $BUILD_DIRECTORY
        wget https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-3.5.1.tar.gz && tar zxvf llvmorg-3.5.1.tar.gz
        cd ./llvm-project-llvmorg-3.5.1/llvm && mkdir build && cd build && cmake -DLLVM_REQUIRES_RTTI=1 .. && make -j4 && sudo make install || exit -1
        cd $BUILD_DIRECTORY
        git clone --recursive https://github.com/microsoft/bond.git
        sudo DEBIAN_FRONTEND="noninteractive" apt-get install -y clang zlib1g-dev
        sudo curl -sSL https://get.haskellstack.org/ | sh
        cd ./bond && mkdir build && cd build && cmake -DBOND_ENABLE_GRPC=FALSE .. && make -j4 && sudo make install || exit -1
    fi
    ARCH=$(uname -m)
    if [[ $TASK == "cuda" ]]; then
        echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
        apt-get update
        apt-get install --no-install-recommends -y \
            curl \
            graphviz \
            libxau6 \
            libxext6 \
            libxrender1 \
            lsb-release \
            software-properties-common
        if [[ $COMPILER == "clang" ]]; then
            apt-get install --no-install-recommends -y \
                clang \
                libomp-dev
        fi
        curl \
            -s \
            -L \
            --insecure \
            https://apt.kitware.com/keys/kitware-archive-latest.asc \
        | apt-key add -
        apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" -y
        apt-get --allow-unauthenticated upgrade -y
        apt-get --allow-unauthenticated update -y
        apt-get install --no-install-recommends -y \
            cmake
    else
        if [[ $ARCH != "x86_64" ]]; then
            yum update -y
            yum install -y \
                graphviz
        else
            sudo apt-get update
            sudo apt-get install --no-install-recommends -y \
                graphviz
        fi
    fi
    if [[ $SETUP_CONDA != "false" ]]; then
        if [[ $ARCH == "x86_64" ]]; then
            curl -sL -o conda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        else
            curl -sL -o conda.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${ARCH}.sh
        fi
    fi
fi

if [[ "${TASK}" != "r-package" ]]; then
    if [[ $SETUP_CONDA != "false" ]]; then
        sh conda.sh -b -p $CONDA
    fi
    conda config --set always_yes yes --set changeps1 no
    conda update -q -y conda
fi
