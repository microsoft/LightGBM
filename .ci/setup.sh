#!/bin/bash


echo "--- start ---"
echo 45


BUILD_ARGS=""
BUILD_ARGS="${BUILD_ARGS} --config-setting=cmake.define.CMAKE_SH=CMAKE_SH-NOTFOUND --config-setting=cmake.args=-G'MinGW Makefiles'"


echo "${BUILD_ARGS}" \
    | xargs -L1 \
        printf \
            "%s\n"

# echo "${BUILD_ARGS}" \
#     | xargs \
#         python -m build \
#             --wheel \
#             --outdir ../dist \
#             .



echo "--- end ---"







# echo 61
# echo "$SHELL"
# xargs --version

# ARCH=$(uname -m)


# export SKBUILD_LOGGING_LEVEL="INFO"

# # brew install libomp  # open-mpi
# # export CXX=g++-14 CC=gcc-14

# # sudo apt-get update
# # sudo apt-get install --no-install-recommends -y \
# #     libboost1.74-dev \
# #     libboost-filesystem1.74-dev
#     # ocl-icd-opencl-dev
#     # pocl-opencl-icd

# # mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# curl \
#     -sL \
#     -o miniforge.sh \
#     "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-${ARCH}.sh"
# sh miniforge.sh -b -p "${CONDA}"
# conda config --set always_yes yes --set changeps1 no
# conda update -q -y conda


# pip install pytest numpy pandas scipy scikit-learn psutil cloudpickle


# cd $GITHUB_WORKSPACE/.ci
# git clone --recursive -b docs/install-py https://github.com/microsoft/LightGBM.git
# cd LightGBM
# sh ./build-python.sh install --nomp


# pytest ./tests/python_package_test || exit 1
