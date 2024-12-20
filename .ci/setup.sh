#!/bin/bash



ARCH=$(uname -m)


# brew install libomp open-mpi
# export CXX=g++-14 CC=gcc-14

curl \
    -sL \
    -o miniforge.sh \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-${ARCH}.sh"
sh miniforge.sh -b -p "${CONDA}"
conda config --set always_yes yes --set changeps1 no
conda update -q -y conda


pip install pytest numpy pandas scipy scikit-learn psutil cloudpickle
pip install lightgbm --no-binary lightgbm --config-settings=cmake.define.USE_OPENMP=OFF

cd "${BUILD_DIRECTORY}"
pytest ./tests/python_package_test || exit 1



# else  # Linux
#     if type -f apt > /dev/null 2>&1; then
#         sudo apt-get update
#         sudo apt-get install --no-install-recommends -y \
#             ca-certificates \
#             curl
#     else
#         sudo yum update -y
#         sudo yum install -y \
#             ca-certificates \
#             curl
#     fi
#     CMAKE_VERSION="3.30.0"
#     curl -O -L \
#         "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-${ARCH}.sh" \
#     || exit 1
#     sudo mkdir /opt/cmake || exit 1
#     sudo sh "cmake-${CMAKE_VERSION}-linux-${ARCH}.sh" --skip-license --prefix=/opt/cmake || exit 1
#     sudo ln -sf /opt/cmake/bin/cmake /usr/local/bin/cmake || exit 1

#     if [[ $IN_UBUNTU_BASE_CONTAINER == "true" ]]; then
#         # fixes error "unable to initialize frontend: Dialog"
#         # https://github.com/moby/moby/issues/27988#issuecomment-462809153
#         echo 'debconf debconf/frontend select Noninteractive' | sudo debconf-set-selections

#         sudo apt-get update
#         sudo apt-get install --no-install-recommends -y \
#             software-properties-common

#         sudo apt-get install --no-install-recommends -y \
#             build-essential \
#             git \
#             libcurl4 \
#             libicu-dev \
#             libssl-dev \
#             locales \
#             locales-all || exit 1
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
#     if [[ $TASK == "mpi" ]]; then
#         if [[ $IN_UBUNTU_BASE_CONTAINER == "true" ]]; then
#             sudo apt-get update
#             sudo apt-get install --no-install-recommends -y \
#                 libopenmpi-dev \
#                 openmpi-bin
#         else  # in manylinux image
#             sudo yum update -y
#             sudo yum install -y \
#                 openmpi-devel \
#             || exit 1
#         fi
#     fi
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
