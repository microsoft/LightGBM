#!/bin/bash

if [[ $OS_NAME == "macos" ]]; then
    if  [[ $COMPILER == "clang" ]]; then
        brew install libomp
        if [[ $AZURE == "true" ]]; then
            sudo xcode-select -s /Applications/Xcode_8.3.3.app/Contents/Developer
        fi
    else  # gcc
        if [[ $TRAVIS == "true" ]]; then
#            rm '/usr/local/include/c++'  # previous variant to deal with conflict link
#            brew cask uninstall oclint  #  reserve variant to deal with conflict link
            brew link --overwrite gcc
            brew upgrade gcc
        else
            brew update
        fi
        if [[ $TASK != "mpi" ]]; then
            brew install gcc
        fi
    fi
    if [[ $TASK == "mpi" ]]; then
        brew install open-mpi
    fi
    if [[ $AZURE == "true" ]] && [[ $TASK == "sdist" ]]; then
        brew install https://raw.githubusercontent.com/Homebrew/homebrew-core/f3544543a3115023fc7ca962c21d14b443f419d0/Formula/swig.rb  # swig 3.0.12
    fi
    wget -q -O conda.sh https://repo.continuum.io/miniconda/Miniconda${PYTHON_VERSION:0:1}-latest-MacOSX-x86_64.sh
else  # Linux
    if [[ $AZURE == "true" ]] && [[ $COMPILER == "clang" ]]; then
        sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-7 100
        sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-7 100
    fi
    if [[ $TASK == "mpi" ]]; then
        sudo apt-get update
        sudo apt-get install --no-install-recommends -y libopenmpi-dev openmpi-bin
    fi
    if [[ $TASK == "gpu" ]]; then
        sudo add-apt-repository ppa:mhier/libboost-latest -y
        sudo apt-get update
        sudo apt-get install --no-install-recommends -y libboost1.70-dev ocl-icd-opencl-dev
        cd $BUILD_DIRECTORY  # to avoid permission errors
        wget -q https://github.com/microsoft/LightGBM/releases/download/v2.0.12/AMD-APP-SDKInstaller-v3.0.130.136-GA-linux64.tar.bz2
        tar -xjf AMD-APP-SDK*.tar.bz2
        mkdir -p $OPENCL_VENDOR_PATH
        mkdir -p $AMDAPPSDK_PATH
        sh AMD-APP-SDK*.sh --tar -xf -C $AMDAPPSDK_PATH
        mv $AMDAPPSDK_PATH/lib/x86_64/sdk/* $AMDAPPSDK_PATH/lib/x86_64/
        echo libamdocl64.so > $OPENCL_VENDOR_PATH/amdocl64.icd
    fi
    if [[ $TRAVIS == "true" ]]; then
        wget -q -O conda.sh https://repo.continuum.io/miniconda/Miniconda${PYTHON_VERSION:0:1}-latest-Linux-x86_64.sh
    fi
fi

if [[ $TRAVIS == "true" ]] || [[ $OS_NAME == "macos" ]]; then
    sh conda.sh -b -p $CONDA
fi
conda config --set always_yes yes --set changeps1 no
conda update -q -y conda
