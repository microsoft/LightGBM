#!/bin/bash

set -e -E -o -u pipefail

# defaults
CONDA_ENV="test-env"
IN_UBUNTU_BASE_CONTAINER=${IN_UBUNTU_BASE_CONTAINER:-"false"}
METHOD=${METHOD:-""}
PRODUCES_ARTIFACTS=${PRODUCES_ARTIFACTS:-"false"}
SANITIZERS=${SANITIZERS:-""}

ARCH=$(uname -m)

LGB_VER=$(head -n 1 "${BUILD_DIRECTORY}/VERSION.txt")

if [[ $OS_NAME == "macos" ]] && [[ $COMPILER == "gcc" ]]; then
    export CXX=g++-12
    export CC=gcc-12
elif [[ $OS_NAME == "linux" ]] && [[ $COMPILER == "clang" ]]; then
    export CXX=clang++
    export CC=clang
elif [[ $OS_NAME == "linux" ]] && [[ $COMPILER == "clang-17" ]]; then
    export CXX=clang++-17
    export CC=clang-17
fi

if [[ $IN_UBUNTU_BASE_CONTAINER == "true" ]]; then
    export LANG="en_US.UTF-8"
    export LC_ALL="en_US.UTF-8"
fi

# Setting MACOSX_DEPLOYMENT_TARGET prevents CMake from building against too-new
# macOS features, and helps tools like Python build tools determine the appropriate
# wheel compatibility tags.
#
# ref:
#   * https://cmake.org/cmake/help/latest/envvar/MACOSX_DEPLOYMENT_TARGET.html
#   * https://github.com/scikit-build/scikit-build-core/blob/acb7d0346e4a05bcb47a4ea3939c705ab71e3145/src/scikit_build_core/builder/macos.py#L36
if [[ $ARCH == "x86_64" ]]; then
    export MACOSX_DEPLOYMENT_TARGET=10.15
else
    export MACOSX_DEPLOYMENT_TARGET=12.0
fi

if [[ "${TASK}" == "r-package" ]]; then
    bash "${BUILD_DIRECTORY}/.ci/test-r-package.sh" || exit 1
    exit 0
fi



# including python=version[build=*cpython] to ensure that conda doesn't fall back to pypy
CONDA_PYTHON_REQUIREMENT="python=${PYTHON_VERSION}[build=*cpython]"

cd "${BUILD_DIRECTORY}"


if [[ $PYTHON_VERSION == "3.7" ]]; then
    CONDA_REQUIREMENT_FILE="${BUILD_DIRECTORY}/.ci/conda-envs/ci-core-py37.txt"
elif [[ $PYTHON_VERSION == "3.8" ]]; then
    CONDA_REQUIREMENT_FILE="${BUILD_DIRECTORY}/.ci/conda-envs/ci-core-py38.txt"
else
    CONDA_REQUIREMENT_FILE="${BUILD_DIRECTORY}/.ci/conda-envs/ci-core.txt"
fi

conda create \
    -y \
    -n "${CONDA_ENV}" \
    --file "${CONDA_REQUIREMENT_FILE}" \
    "${CONDA_PYTHON_REQUIREMENT}" \
|| exit 1

# shellcheck disable=SC1091
source activate $CONDA_ENV

cd "${BUILD_DIRECTORY}"

if [[ $TASK == "sdist" ]]; then
    sh ./build-python.sh sdist || exit 1
    sh .ci/check-python-dists.sh ./dist || exit 1
    pip install "./dist/lightgbm-${LGB_VER}.tar.gz" -v || exit 1
    if [[ $PRODUCES_ARTIFACTS == "true" ]]; then
        cp "./dist/lightgbm-${LGB_VER}.tar.gz" "${BUILD_ARTIFACTSTAGINGDIRECTORY}" || exit 1
    fi
    pytest ./tests/python_package_test || exit 1
    exit 0
elif [[ $TASK == "bdist" ]]; then
    if [[ $OS_NAME == "macos" ]]; then
        sh ./build-python.sh bdist_wheel || exit 1
        sh .ci/check-python-dists.sh ./dist || exit 1
        if [[ $PRODUCES_ARTIFACTS == "true" ]]; then
            cp "$(echo "dist/lightgbm-${LGB_VER}-py3-none-macosx"*.whl)" "${BUILD_ARTIFACTSTAGINGDIRECTORY}" || exit 1
        fi
    else
        if [[ $ARCH == "x86_64" ]]; then
            PLATFORM="manylinux_2_28_x86_64"
        else
            PLATFORM="manylinux2014_$ARCH"
        fi
        sh ./build-python.sh bdist_wheel --integrated-opencl || exit 1
        # rename wheel, to fix scikit-build-core choosing the platform 'linux_aarch64' instead of
        # a manylinux tag
        mv \
            ./dist/*.whl \
            ./dist/tmp.whl || exit 1
        mv \
            ./dist/tmp.whl \
            "./dist/lightgbm-${LGB_VER}-py3-none-${PLATFORM}.whl" || exit 1
        sh .ci/check-python-dists.sh ./dist || exit 1
        if [[ $PRODUCES_ARTIFACTS == "true" ]]; then
            cp "dist/lightgbm-${LGB_VER}-py3-none-${PLATFORM}.whl" "${BUILD_ARTIFACTSTAGINGDIRECTORY}" || exit 1
        fi
        # Make sure we can do both CPU and GPU; see tests/python_package_test/test_dual.py
        export LIGHTGBM_TEST_DUAL_CPU_GPU=1
    fi
    pip install -v ./dist/*.whl || exit 1
    pytest ./tests || exit 1
    exit 0
fi

if [[ $TASK == "gpu" ]]; then
    sed -i'.bak' 's/std::string device_type = "cpu";/std::string device_type = "gpu";/' ./include/LightGBM/config.h
    grep -q 'std::string device_type = "gpu"' ./include/LightGBM/config.h || exit 1  # make sure that changes were really done
    if [[ $METHOD == "pip" ]]; then
        sh ./build-python.sh sdist || exit 1
        sh .ci/check-python-dists.sh ./dist || exit 1
        pip install \
            -v \
            --config-settings=cmake.define.USE_GPU=ON \
            "./dist/lightgbm-${LGB_VER}.tar.gz" \
        || exit 1
        pytest ./tests/python_package_test || exit 1
        exit 0
    elif [[ $METHOD == "wheel" ]]; then
        sh ./build-python.sh bdist_wheel --gpu || exit 1
        sh ./.ci/check-python-dists.sh ./dist || exit 1
        pip install "$(echo "./dist/lightgbm-${LGB_VER}"*.whl)" -v || exit 1
        pytest ./tests || exit 1
        exit 0
    elif [[ $METHOD == "source" ]]; then
        cmake -B build -S . -DUSE_GPU=ON
    fi
elif [[ $TASK == "cuda" ]]; then
    sed -i'.bak' 's/std::string device_type = "cpu";/std::string device_type = "cuda";/' ./include/LightGBM/config.h
    grep -q 'std::string device_type = "cuda"' ./include/LightGBM/config.h || exit 1  # make sure that changes were really done
    # by default ``gpu_use_dp=false`` for efficiency. change to ``true`` here for exact results in ci tests
    sed -i'.bak' 's/gpu_use_dp = false;/gpu_use_dp = true;/' ./include/LightGBM/config.h
    grep -q 'gpu_use_dp = true' ./include/LightGBM/config.h || exit 1  # make sure that changes were really done
    if [[ $METHOD == "pip" ]]; then
        sh ./build-python.sh sdist || exit 1
        sh ./.ci/check-python-dists.sh ./dist || exit 1
        pip install \
            -v \
            --config-settings=cmake.define.USE_CUDA=ON \
            "./dist/lightgbm-${LGB_VER}.tar.gz" \
        || exit 1
        pytest ./tests/python_package_test || exit 1
        exit 0
    elif [[ $METHOD == "wheel" ]]; then
        sh ./build-python.sh bdist_wheel --cuda || exit 1
        sh ./.ci/check-python-dists.sh ./dist || exit 1
        pip install "$(echo "./dist/lightgbm-${LGB_VER}"*.whl)" -v || exit 1
        pytest ./tests || exit 1
        exit 0
    elif [[ $METHOD == "source" ]]; then
        cmake -B build -S . -DUSE_CUDA=ON
    fi
elif [[ $TASK == "mpi" ]]; then
    if [[ $METHOD == "pip" ]]; then
        sh ./build-python.sh sdist || exit 1
        sh ./.ci/check-python-dists.sh ./dist || exit 1
        pip install \
            -v \
            --config-settings=cmake.define.USE_MPI=ON \
            "./dist/lightgbm-${LGB_VER}.tar.gz" \
        || exit 1
        pytest ./tests/python_package_test || exit 1
        exit 0
    elif [[ $METHOD == "wheel" ]]; then
        sh ./build-python.sh bdist_wheel --mpi || exit 1
        sh ./.ci/check-python-dists.sh ./dist || exit 1
        pip install "$(echo "./dist/lightgbm-${LGB_VER}"*.whl)" -v || exit 1
        pytest ./tests || exit 1
        exit 0
    elif [[ $METHOD == "source" ]]; then
        cmake -B build -S . -DUSE_MPI=ON -DUSE_DEBUG=ON
    fi
else
    cmake -B build -S .
fi

cmake --build build --target _lightgbm -j4 || exit 1

sh ./build-python.sh install --precompile || exit 1
