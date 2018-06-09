#!/bin/bash

if [[ $TRAVIS_OS_NAME == "osx" ]]; then
    rm '/usr/local/include/c++'
#    brew cask uninstall oclint  #  Reserve variant to deal with conflict link
    if [[ ${TASK} == "mpi" ]]; then
        brew install open-mpi
    else
        brew install gcc
    fi
#    brew link --overwrite gcc  # Previous variant to deal with conflict link
    wget -O conda.sh https://repo.continuum.io/miniconda/Miniconda${PYTHON_VERSION:0:1}-latest-MacOSX-x86_64.sh
else
    if [[ ${TASK} == "mpi" ]]; then
        sudo apt-get install -y libopenmpi-dev openmpi-bin
    fi
    if [[ ${TASK} == "gpu" ]]; then
        sudo apt-get install -y ocl-icd-opencl-dev
    fi
    wget -O conda.sh https://repo.continuum.io/miniconda/Miniconda${PYTHON_VERSION:0:1}-latest-Linux-x86_64.sh
fi
