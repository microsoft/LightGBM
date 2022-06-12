#!/bin/bash

if [[ $OS_NAME == "macos" ]]; then
    if  [[ $COMPILER == "clang" ]]; then
        brew install libomp
        if [[ $AZURE == "true" ]]; then
            sudo xcode-select -s /Applications/Xcode_10.3.app/Contents/Developer || exit -1
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
    curl \
        -sL \
        -o miniforge.sh \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh
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
        if [[ $IN_UBUNTU_LATEST_CONTAINER == "true" ]]; then
            sudo apt-get install --no-install-recommends -y \
                pocl-opencl-icd
        else
            sudo apt-get install --no-install-recommends -y \
                libhwloc-dev \
                libtinfo-dev \
                ocl-icd-dev \
                pkg-config \
                zlib1g-dev
            git clone --depth 1 --branch v1.8 https://github.com/pocl/pocl.git
            cmake -B pocl/build -S pocl -DCMAKE_BUILD_TYPE=release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_FLAGS=-stdlib=libc++ -DPOCL_INSTALL_ICD_VENDORDIR=/etc/OpenCL/vendors -DPOCL_DEBUG_MESSAGES=OFF -DSTATIC_LLVM=ON -DINSTALL_OPENCL_HEADERS=OFF -DENABLE_SPIR=OFF -DENABLE_POCLCC=OFF -DENABLE_TESTS=OFF -DENABLE_EXAMPLES=OFF
            cmake --build pocl/build -j4
            sudo cmake --install pocl/build
        fi
    fi
    if [[ $TASK == "cuda" || $TASK == "cuda_exp" ]]; then
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
        ARCH=$(uname -m)
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
