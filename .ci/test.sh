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

if [[ "$TASK" == "cpp-tests" ]]; then
    cmake_args=(
        -DBUILD_CPP_TEST=ON
        -DUSE_DEBUG=ON
    )
    if [[ $METHOD == "with-sanitizers" ]]; then
        cmake_args+=("-DUSE_SANITIZER=ON")
        if [[ -n $SANITIZERS ]]; then
            cmake_args+=("-DENABLED_SANITIZERS=$SANITIZERS")
        fi
    fi
    cmake -B build -S . "${cmake_args[@]}"
    cmake --build build --target testlightgbm -j4 || exit 1
    ./testlightgbm || exit 1
    exit 0
fi

# including python=version=[build=*_cp*] to ensure that conda prefers CPython and doesn't fall back to
# other implementations like pypy
CONDA_PYTHON_REQUIREMENT="python=${PYTHON_VERSION}[build=*_cp*]"

if [[ $TASK == "if-else" ]]; then
    conda create -q -y -n "${CONDA_ENV}" "${CONDA_PYTHON_REQUIREMENT}" numpy
    # shellcheck disable=SC1091
    source activate "${CONDA_ENV}"
    cmake -B build -S . || exit 1
    cmake --build build --target lightgbm -j4 || exit 1
    cd "$BUILD_DIRECTORY/tests/cpp_tests"
    ../../lightgbm config=train.conf convert_model_language=cpp convert_model=../../src/boosting/gbdt_prediction.cpp
    ../../lightgbm config=predict.conf output_result=origin.pred
    ../../lightgbm config=predict.conf output_result=ifelse.pred
    python test.py
    exit 0
fi

cd "${BUILD_DIRECTORY}"

if [[ $TASK == "swig" ]]; then
    cmake -B build -S . -DUSE_SWIG=ON
    cmake --build build -j4 || exit 1
    if [[ $OS_NAME == "linux" ]] && [[ $COMPILER == "gcc" ]]; then
        objdump -T ./lib_lightgbm.so > ./objdump.log || exit 1
        objdump -T ./lib_lightgbm_swig.so >> ./objdump.log || exit 1
        python ./.ci/check-dynamic-dependencies.py ./objdump.log || exit 1
    fi
    if [[ $PRODUCES_ARTIFACTS == "true" ]]; then
        cp ./build/lightgbmlib.jar "${BUILD_ARTIFACTSTAGINGDIRECTORY}/lightgbmlib_${OS_NAME}.jar"
    fi
    exit 0
fi

if [[ $TASK == "lint" ]]; then
    pwsh -command "Install-Module -Name PSScriptAnalyzer -Scope CurrentUser -SkipPublisherCheck"
    echo "Linting PowerShell code"
    pwsh -file ./.ci/lint-powershell.ps1 || exit 1
    conda create -q -y -n "${CONDA_ENV}" \
        "${CONDA_PYTHON_REQUIREMENT}" \
        'biome>=1.9.3' \
        'cmakelint>=1.4.3' \
        'cpplint>=1.6.0' \
        'matplotlib-base>=3.9.1' \
        'mypy>=1.11.1' \
        'pre-commit>=3.8.0' \
        'pyarrow-core>=17.0' \
        'scikit-learn>=1.5.2' \
        'r-lintr>=3.1.2'
    # shellcheck disable=SC1091
    source activate "${CONDA_ENV}"
    echo "Linting Python and bash code"
    bash ./.ci/lint-python-bash.sh || exit 1
    echo "Linting R code"
    Rscript ./.ci/lint-r-code.R "${BUILD_DIRECTORY}" || exit 1
    echo "Linting C++ code"
    bash ./.ci/lint-cpp.sh || exit 1
    echo "Linting JavaScript code"
    bash ./.ci/lint-js.sh || exit 1
    exit 0
fi

if [[ $TASK == "check-docs" ]] || [[ $TASK == "check-links" ]]; then
    conda env create \
        -n "${CONDA_ENV}" \
        --file ./docs/env.yml || exit 1
    conda install \
        -q \
        -y \
        -n "${CONDA_ENV}" \
            'doxygen>=1.10.0' \
            'rstcheck>=6.2.4' || exit 1
    # shellcheck disable=SC1091
    source activate "${CONDA_ENV}"
    # check reStructuredText formatting
    find "${BUILD_DIRECTORY}/python-package" -type f -name "*.rst" \
        -exec rstcheck --report-level warning {} \+ || exit 1
    find "${BUILD_DIRECTORY}/docs" -type f -name "*.rst" \
        -exec rstcheck --report-level warning --ignore-directives=autoclass,autofunction,autosummary,doxygenfile {} \+ || exit 1
    # build docs
    make -C docs html || exit 1
    if [[ $TASK == "check-links" ]]; then
        # check docs for broken links
        pip install 'linkchecker>=10.5.0'
        linkchecker --config=./docs/.linkcheckerrc ./docs/_build/html/*.html || exit 1
        exit 0
    fi
    # check the consistency of parameters' descriptions and other stuff
    cp ./docs/Parameters.rst ./docs/Parameters-backup.rst
    cp ./src/io/config_auto.cpp ./src/io/config_auto-backup.cpp
    python ./.ci/parameter-generator.py || exit 1
    diff ./docs/Parameters-backup.rst ./docs/Parameters.rst || exit 1
    diff ./src/io/config_auto-backup.cpp ./src/io/config_auto.cpp || exit 1
    exit 0
fi

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
pytest ./tests || exit 1

if [[ $TASK == "regular" ]]; then
    if [[ $PRODUCES_ARTIFACTS == "true" ]]; then
        if [[ $OS_NAME == "macos" ]]; then
            cp ./lib_lightgbm.dylib "${BUILD_ARTIFACTSTAGINGDIRECTORY}/lib_lightgbm.dylib"
        else
            if [[ $COMPILER == "gcc" ]]; then
                objdump -T ./lib_lightgbm.so > ./objdump.log || exit 1
                python ./.ci/check-dynamic-dependencies.py ./objdump.log || exit 1
            fi
            cp ./lib_lightgbm.so "${BUILD_ARTIFACTSTAGINGDIRECTORY}/lib_lightgbm.so"
        fi
    fi
    cd "$BUILD_DIRECTORY/examples/python-guide"
    sed -i'.bak' '/import lightgbm as lgb/a\
import matplotlib\
matplotlib.use\(\"Agg\"\)\
' plot_example.py  # prevent interactive window mode
    sed -i'.bak' 's/graph.render(view=True)/graph.render(view=False)/' plot_example.py
    # requirements for examples
    conda install -y -n $CONDA_ENV \
        'h5py>=3.10' \
        'ipywidgets>=8.1.2' \
        'notebook>=7.1.2'
    for f in *.py **/*.py; do python "${f}" || exit 1; done  # run all examples
    cd "$BUILD_DIRECTORY/examples/python-guide/notebooks"
    sed -i'.bak' 's/INTERACTIVE = False/assert False, \\"Interactive mode disabled\\"/' interactive_plot_example.ipynb
    jupyter nbconvert --ExecutePreprocessor.timeout=180 --to notebook --execute --inplace ./*.ipynb || exit 1  # run all notebooks

    # importing the library should succeed even if all optional dependencies are not present
    conda uninstall -n $CONDA_ENV --force --yes \
        cffi \
        dask \
        distributed \
        joblib \
        matplotlib-base \
        pandas \
        psutil \
        pyarrow \
        python-graphviz \
        scikit-learn || exit 1
    python -c "import lightgbm" || exit 1
fi
