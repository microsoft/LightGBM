#!/bin/bash

if [[ $OS_NAME == "macos" ]]; then
    if  [[ $COMPILER == "clang" ]]; then
        brew install libomp
        brew upgrade cmake  # CMake >=3.12 is needed to find OpenMP at macOS
        if [[ $AZURE == "true" ]]; then
            sudo xcode-select -s /Applications/Xcode_8.3.1.app/Contents/Developer
        fi
    else
        if [[ $TRAVIS == "true" ]]; then
#            rm '/usr/local/include/c++'  # previous variant to deal with conflict link
#            brew cask uninstall oclint  #  reserve variant to deal with conflict link
            brew link --overwrite gcc
        fi
        if [[ $TASK != "mpi" ]]; then
            brew install gcc
        fi
    fi
    if [[ $TASK == "mpi" ]]; then
        brew install open-mpi
    fi
    if [[ $TRAVIS == "true" ]]; then
        wget -O conda.sh https://repo.continuum.io/miniconda/Miniconda${PYTHON_VERSION:0:1}-latest-MacOSX-x86_64.sh
    fi
else  # Linux
    if [[ $AZURE == "true" ]] && [[ $COMPILER == "clang" ]]; then
        sudo apt-get update
        sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-6.0 100
        sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-6.0 100
        sudo apt-get install --no-install-recommends -y libomp-dev
    fi
    if [[ $TASK == "mpi" ]]; then
        sudo apt-get update
        sudo apt-get install --no-install-recommends -y libopenmpi-dev openmpi-bin
    fi
    if [[ $TASK == "gpu" ]]; then
        if [[ $TRAVIS == "true" ]]; then
            sudo add-apt-repository ppa:kzemek/boost -y
        fi
        sudo apt-get update
        sudo apt-get install --no-install-recommends -y libboost1.58-dev libboost-system1.58-dev libboost-filesystem1.58-dev ocl-icd-opencl-dev
        cd $HOME_DIRECTORY
        wget -q https://github.com/Microsoft/LightGBM/releases/download/v2.0.12/AMD-APP-SDKInstaller-v3.0.130.136-GA-linux64.tar.bz2
        tar -xjf AMD-APP-SDK*.tar.bz2
        mkdir -p $OPENCL_VENDOR_PATH
        mkdir -p $AMDAPPSDK_PATH
        sh AMD-APP-SDK*.sh --tar -xf -C $AMDAPPSDK_PATH
        mv $AMDAPPSDK_PATH/lib/x86_64/sdk/* $AMDAPPSDK_PATH/lib/x86_64/
        echo libamdocl64.so > $OPENCL_VENDOR_PATH/amdocl64.icd
    fi
    if [[ $TRAVIS == "true" ]]; then
        wget -O conda.sh https://repo.continuum.io/miniconda/Miniconda${PYTHON_VERSION:0:1}-latest-Linux-x86_64.sh
    fi
fi

if [[ $TRAVIS == "true" ]]; then
    sh conda.sh -b -p $HOME_DIRECTORY/miniconda
fi
conda config --set always_yes yes --set changeps1 no
conda update -q conda
