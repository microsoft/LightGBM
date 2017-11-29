#!/bin/bash

if [[ $TRAVIS_OS_NAME == "osx" ]]; then
    if [[ ${TASK} == "mpi" ]]; then
        brew install open-mpi
    else
        brew install gcc
    fi
    brew link --overwrite gcc
    wget -O conda.sh https://repo.continuum.io/miniconda/Miniconda${PYTHON_VERSION:0:1}-latest-MacOSX-x86_64.sh
else
    if [[ ${TASK} != "pylint" ]] && [[ ${TASK} != "check-docs" ]]; then
        sudo add-apt-repository ppa:george-edison55/cmake-3.x -y
        sudo apt-get update -q
        sudo apt-get install -y cmake
        sudo apt-get install -y libopenmpi-dev openmpi-bin build-essential
    fi
    wget -O conda.sh https://repo.continuum.io/miniconda/Miniconda${PYTHON_VERSION:0:1}-latest-Linux-x86_64.sh
fi
