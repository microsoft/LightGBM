#!/bin/bash

#sudo apt-get update 
#sudo apt-get install --no-install-recommends -y \
#    clang \
#    libomp-dev

ARCH="x86_64"
CMAKE_VERSION="3.30.0"
curl -O -L https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-${ARCH}.sh
sudo mkdir /opt/cmake
sudo sh cmake-${CMAKE_VERSION}-linux-${ARCH}.sh --skip-license --prefix=/opt/cmake
sudo ln -sf /opt/cmake/bin/cmake /usr/local/bin/cmake
cmake --version

curl -O -L https://github.com/ninja-build/ninja/releases/download/v1.12.1/ninja-linux.zip
unzip ninja-linux.zip -d /usr/local/bin/
ninja --version

sudo apt-get update
sudo apt-get install --no-install-recommends -y \
    libboost1.74-dev \
    libboost-filesystem1.74-dev \
    ocl-icd-opencl-dev

git clone --recursive https://github.com/microsoft/LightGBM
cd LightGBM
cmake -B build -S . -DUSE_GPU=ON
# if you have installed NVIDIA CUDA to a customized location, you should specify paths to OpenCL headers and library like the following:
# cmake -B build -S . -DUSE_GPU=ON -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/
cmake --build build

ls .

cd ./examples/regression
../../lightgbm config=train.conf

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
