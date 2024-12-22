#!/bin/bash

echo 59


ARCH=$(uname -m)


export SKBUILD_LOGGING_LEVEL="INFO"

# brew install libomp open-mpi
# export CXX=g++-14 CC=gcc-14

sudo apt-get update
sudo apt-get install --no-install-recommends -y \
    libboost1.74-dev \
    libboost-filesystem1.74-dev
    # ocl-icd-opencl-dev
    # pocl-opencl-icd

# mkdir -p /etc/OpenCL/vendors && echo "libOpenCL.so" > /etc/OpenCL/vendors/opencl.icd

curl \
    -sL \
    -o miniforge.sh \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-${ARCH}.sh"
sh miniforge.sh -b -p "${CONDA}"
conda config --set always_yes yes --set changeps1 no
conda update -q -y conda


pip install pytest numpy pandas scipy scikit-learn psutil cloudpickle



git clone --recursive -b ci/test https://github.com/microsoft/LightGBM.git
cd LightGBM
sh ./build-python.sh install --gpu


pytest ./tests/python_package_test || exit 1




#     if [[ $IN_UBUNTU_BASE_CONTAINER == "true" ]]; then
#         # fixes error "unable to initialize frontend: Dialog"
#         # https://github.com/moby/moby/issues/27988#issuecomment-462809153
#         echo 'debconf debconf/frontend select Noninteractive' | sudo debconf-set-selections

#         if [[ $COMPILER == "clang" ]]; then
#             sudo apt-get install --no-install-recommends -y \
#                 clang \
#                 libomp-dev
#         elif [[ $COMPILER == "clang-17" ]]; then
#             sudo apt-get install --no-install-recommends -y \
#                 wget
#             wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | sudo tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc
#             sudo apt-add-repository deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-17 main
#             sudo apt-add-repository deb-src http://apt.llvm.org/jammy/ llvm-toolchain-jammy-17 main
#             sudo apt-get update
#             sudo apt-get install -y \
#                 clang-17 \
#                 libomp-17-dev
#         fi

#         export LANG="en_US.UTF-8"
#         sudo update-locale LANG=${LANG}
#         export LC_ALL="${LANG}"
#     fi
#     if [[ $TASK == "gpu" ]]; then
#         if [[ $IN_UBUNTU_BASE_CONTAINER == "true" ]]; then

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
