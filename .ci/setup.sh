#!/bin/bash

if [[ $OS_NAME == "macos" ]]; then
    if  [[ $COMPILER == "clang" ]]; then
        brew install libomp
        brew reinstall cmake  # CMake >=3.12 is needed to find OpenMP at macOS
    else
        if [[ $TRAVIS == "true" ]]; then
            sudo softwareupdate -i "Command Line Tools (macOS High Sierra version 10.13) for Xcode-9.3"  # fix "fatal error: _stdio.h: No such file or directory"
            rm '/usr/local/include/c++'
#            brew cask uninstall oclint  #  reserve variant to deal with conflict link
#            brew link --overwrite gcc  # previous variant to deal with conflict link
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
    sudo apt-get update
    if [[ $AZURE == "true" ]] && [[ $COMPILER == "clang" ]]; then
        sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-6.0 100
        sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-6.0 100
        sudo apt-get install libomp-dev
    elif [[ $AZURE == "true" ]] && [[ $COMPILER == "gcc" ]] && [[ $TASK != "gpu" ]]; then
        # downgrade gcc version
        sudo apt-get remove -y gcc || exit -1
        sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
        sudo apt-get update
        sudo apt-get install --no-install-recommends -y g++-4.8 || exit -1
        sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 100
        sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 100
    fi
    if [[ $TASK == "mpi" ]]; then
        sudo apt-get install --no-install-recommends -y libopenmpi-dev openmpi-bin
    fi
    if [[ $TASK == "gpu" ]]; then
        if [[ $AZURE == "true" ]]; then
            sudo apt-get install --no-install-recommends -y libboost-dev libboost-system-dev libboost-filesystem-dev
        fi
        sudo apt-get install --no-install-recommends -y ocl-icd-opencl-dev
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
    conda config --set always_yes yes --set changeps1 no
    conda update -q conda
fi
