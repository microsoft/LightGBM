#!/bin/bash

if [[ $TRAVIS_OS_NAME == "osx" ]]; then
    if  [[ $TASK == "clang" ]]; then
        brew reinstall cmake --verbose --HEAD  # CMake >=3.12 is needed
        brew install libomp
    else
        rm '/usr/local/include/c++'
#        brew cask uninstall oclint  #  reserve variant to deal with conflict link
        if [[ $TASK == "mpi" ]]; then
            brew install open-mpi
        else
            brew install gcc
        fi
#        brew link --overwrite gcc  # previous variant to deal with conflict link
    fi
    wget -O conda.sh https://repo.continuum.io/miniconda/Miniconda${PYTHON_VERSION:0:1}-latest-MacOSX-x86_64.sh
else
    if [[ $TASK == "mpi" ]]; then
        sudo apt-get install -y libopenmpi-dev openmpi-bin
    fi
    if [[ $TASK == "gpu" ]]; then
        sudo apt-get install -y ocl-icd-opencl-dev
    fi
    wget -O conda.sh https://repo.continuum.io/miniconda/Miniconda${PYTHON_VERSION:0:1}-latest-Linux-x86_64.sh
fi

sh conda.sh -b -p $HOME/miniconda
conda config --set always_yes yes --set changeps1 no
conda update -q conda

if [[ $TASK == "gpu" ]] && [[ $TRAVIS_OS_NAME == "linux" ]]; then
    wget https://github.com/Microsoft/LightGBM/releases/download/v2.0.12/AMD-APP-SDKInstaller-v3.0.130.136-GA-linux64.tar.bz2
    tar -xjf AMD-APP-SDK*.tar.bz2
    mkdir -p $OPENCL_VENDOR_PATH
    sh AMD-APP-SDK*.sh --tar -xf -C $AMDAPPSDK
    mv $AMDAPPSDK/lib/x86_64/sdk/* $AMDAPPSDK/lib/x86_64/
    echo libamdocl64.so > $OPENCL_VENDOR_PATH/amdocl64.icd
fi
