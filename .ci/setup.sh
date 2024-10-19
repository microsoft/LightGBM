#!/bin/bash

#sudo apt-get update 
#sudo apt-get install --no-install-recommends -y \
#    clang
#    libomp-dev

ARCH="x86_64"
CMAKE_VERSION="3.30.0"
curl -O -L https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-${ARCH}.sh
sudo mkdir /opt/cmake
sudo sh cmake-${CMAKE_VERSION}-linux-${ARCH}.sh --skip-license --prefix=/opt/cmake
sudo ln -sf /opt/cmake/bin/cmake /usr/local/bin/cmake
cmake --version

#curl -O -L https://github.com/ninja-build/ninja/releases/download/v1.12.1/ninja-linux.zip
#unzip ninja-linux.zip -d /usr/local/bin/
#ninja --version

sudo apt-get update
sudo apt-get install --no-install-recommends -y \
    libopenmpi-dev

git clone --recursive https://github.com/microsoft/LightGBM
cd LightGBM
cmake -B build -S . -DUSE_MPI=ON
cmake --build build -j4

ls .

cd ./examples/regression
../../lightgbm config=train.conf

#     if [[ $TASK == "gpu" ]]; then
#         if [[ $IN_UBUNTU_BASE_CONTAINER == "true" ]]; then
#             sudo apt-get update
#             sudo apt-get install --no-install-recommends -y \
#                 libboost1.74-dev \
#                 libboost-filesystem1.74-dev \
#                 ocl-icd-opencl-dev
#         else  # in manylinux image
#             sudo yum update -y
#             sudo yum install -y \
#                 boost-devel \
#                 ocl-icd-devel \
#                 opencl-headers \
#             || exit 1
#         fi
#     fi
#     if [[ $TASK == "gpu" || $TASK == "bdist" ]]; then
#         if [[ $IN_UBUNTU_BASE_CONTAINER == "true" ]]; then
#             sudo apt-get update
#             sudo apt-get install --no-install-recommends -y \
#                 pocl-opencl-icd
#         elif [[ $(uname -m) == "x86_64" ]]; then
#             sudo yum update -y
#             sudo yum install -y \
#                 ocl-icd-devel \
#                 opencl-headers \
#             || exit 1
#         fi
#     fi
#     if [[ $TASK == "cuda" ]]; then
#         echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
#         if [[ $COMPILER == "clang" ]]; then
#             apt-get update
#             apt-get install --no-install-recommends -y \
#                 clang \
#                 libomp-dev
#         fi
#     fi
# fi
