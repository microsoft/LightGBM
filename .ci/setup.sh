#!/bin/bash

set -e -E -u -o pipefail

# defaults
AZURE=${AZURE:-"false"}
IN_UBUNTU_BASE_CONTAINER=${IN_UBUNTU_BASE_CONTAINER:-"false"}
SETUP_CONDA=${SETUP_CONDA:-"true"}

ARCH=$(uname -m)


if [[ $OS_NAME == "macos" ]]; then
    if  [[ $COMPILER == "clang" ]]; then
        brew install libomp
        if [[ $AZURE == "true" ]]; then
            sudo xcode-select -s /Applications/Xcode_11.7.app/Contents/Developer || exit 1
        fi
    else  # gcc
        # Check https://github.com/actions/runner-images/tree/main/images/macos for available
        # versions of Xcode
        sudo xcode-select -s /Applications/Xcode_14.3.1.app/Contents/Developer || exit 1
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
    curl \
        -sL \
        -o miniforge.sh \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-${ARCH}.sh
else  # Linux
    if [[ $IN_UBUNTU_BASE_CONTAINER == "true" ]]; then
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
            libicu-dev \
            libssl-dev \
            libunwind8 \
            locales \
            locales-all \
            netcat \
            unzip \
            zip || exit 1
        if [[ $COMPILER == "clang" ]]; then
            sudo apt-get install --no-install-recommends -y \
                clang \
                libomp-dev
        elif [[ $COMPILER == "clang-17" ]]; then
            sudo apt-get install wget
            wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | sudo tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc
            sudo apt-add-repository deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-17 main
            sudo apt-add-repository deb-src http://apt.llvm.org/jammy/ llvm-toolchain-jammy-17 main
            sudo apt-get update
            sudo apt-get install -y clang-17
            sudo apt-get install --no-install-recommends -y libomp-17-dev
        fi

        export LANG="en_US.UTF-8"
        sudo update-locale LANG=${LANG}
        export LC_ALL="${LANG}"
    fi
    if [[ $TASK == "r-package" ]] && [[ $COMPILER == "clang" ]]; then
        sudo apt-get install --no-install-recommends -y \
            libomp-dev
    fi
    if [[ $TASK == "mpi" ]]; then
        if [[ $IN_UBUNTU_BASE_CONTAINER == "true" ]]; then
            sudo apt-get update
            sudo apt-get install --no-install-recommends -y \
                libopenmpi-dev \
                openmpi-bin
        else  # in manylinux image
            sudo yum update -y
            sudo yum install -y \
                openmpi-devel \
            || exit 1
        fi
    fi
    if [[ $TASK == "gpu" ]]; then
        if [[ $IN_UBUNTU_BASE_CONTAINER == "true" ]]; then
            sudo apt-get update
            sudo apt-get install --no-install-recommends -y \
                libboost1.74-dev \
                libboost-filesystem1.74-dev \
                ocl-icd-opencl-dev
        else  # in manylinux image
            sudo yum update -y
            sudo yum install -y \
                boost-devel \
                ocl-icd-devel \
                opencl-headers \
            || exit 1
        fi
    fi
    if [[ $TASK == "gpu" || $TASK == "bdist" ]]; then
        if [[ $IN_UBUNTU_BASE_CONTAINER == "true" ]]; then
            sudo apt-get update
            sudo apt-get install --no-install-recommends -y \
                pocl-opencl-icd
        elif [[ $(uname -m) == "x86_64" ]]; then
            sudo yum update -y
            sudo yum install -y \
                ocl-icd-devel \
                opencl-headers \
            || exit 1
        fi
    fi
    if [[ $TASK == "cuda" ]]; then
        echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
        apt-get update
        apt-get install --no-install-recommends -y \
            curl \
            lsb-release \
            software-properties-common
        if [[ $COMPILER == "clang" ]]; then
            apt-get install --no-install-recommends -y \
                clang \
                libomp-dev
        fi
        curl -sL https://apt.kitware.com/keys/kitware-archive-latest.asc | apt-key add -
        apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" -y
        apt-get update
        apt-get install --no-install-recommends -y \
            cmake
    fi
    if [[ $SETUP_CONDA != "false" ]]; then
        curl \
            -sL \
            -o miniforge.sh \
            https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${ARCH}.sh
    fi
fi

if [[ "${TASK}" != "r-package" ]] && [[ "${TASK}" != "r-rchk" ]]; then
    if [[ $SETUP_CONDA != "false" ]]; then
        sh miniforge.sh -b -p $CONDA
    fi
    conda config --set always_yes yes --set changeps1 no
    conda update -q -y conda
fi
