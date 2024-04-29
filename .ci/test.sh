#!/bin/bash

set -e -E -o -u pipefail

# defaults
IN_UBUNTU_BASE_CONTAINER=${IN_UBUNTU_BASE_CONTAINER:-"false"}
METHOD=${METHOD:-""}
PRODUCES_ARTIFACTS=${PRODUCES_ARTIFACTS:-"false"}
SANITIZERS=${SANITIZERS:-""}

ARCH=$(uname -m)

if [[ $OS_NAME == "macos" ]] && [[ $COMPILER == "gcc" ]]; then
    export CXX=g++-11
    export CC=gcc-11
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

if [[ "${TASK}" == "r-package" ]] || [[ "${TASK}" == "r-rchk" ]]; then
    bash ${BUILD_DIRECTORY}/.ci/test_r_package.sh || exit 1
    exit 0
fi

if [[ "$TASK" == "cpp-tests" ]]; then
    if [[ $METHOD == "with-sanitizers" ]]; then
        extra_cmake_opts="-DUSE_SANITIZER=ON"
        if [[ -n $SANITIZERS ]]; then
            extra_cmake_opts="$extra_cmake_opts -DENABLED_SANITIZERS=$SANITIZERS"
        fi
    else
        extra_cmake_opts=""
    fi
    cmake -B build -S . -DBUILD_CPP_TEST=ON -DUSE_OPENMP=OFF -DUSE_DEBUG=ON $extra_cmake_opts
    cmake --build build --target testlightgbm -j4 || exit 1
    ./testlightgbm || exit 1
    exit 0
fi

# including python=version[build=*cpython] to ensure that conda doesn't fall back to pypy
CONDA_PYTHON_REQUIREMENT="python=$PYTHON_VERSION[build=*cpython]"

if [[ $TASK == "if-else" ]]; then
    mamba create -q -y -n $CONDA_ENV ${CONDA_PYTHON_REQUIREMENT} numpy
    source activate $CONDA_ENV
    cmake -B build -S . || exit 1
    cmake --build build --target lightgbm -j4 || exit 1
    cd $BUILD_DIRECTORY/tests/cpp_tests && ../../lightgbm config=train.conf convert_model_language=cpp convert_model=../../src/boosting/gbdt_prediction.cpp && ../../lightgbm config=predict.conf output_result=origin.pred || exit 1
    cd $BUILD_DIRECTORY/tests/cpp_tests && ../../lightgbm config=predict.conf output_result=ifelse.pred && python test.py || exit 1
    exit 0
fi

if [[ $TASK == "swig" ]]; then
    cmake -B build -S . -DUSE_SWIG=ON
    cmake --build build -j4 || exit 1
    if [[ $OS_NAME == "linux" ]] && [[ $COMPILER == "gcc" ]]; then
        objdump -T $BUILD_DIRECTORY/lib_lightgbm.so > $BUILD_DIRECTORY/objdump.log || exit 1
        objdump -T $BUILD_DIRECTORY/lib_lightgbm_swig.so >> $BUILD_DIRECTORY/objdump.log || exit 1
        python $BUILD_DIRECTORY/helpers/check_dynamic_dependencies.py $BUILD_DIRECTORY/objdump.log || exit 1
    fi
    if [[ $PRODUCES_ARTIFACTS == "true" ]]; then
        cp $BUILD_DIRECTORY/build/lightgbmlib.jar $BUILD_ARTIFACTSTAGINGDIRECTORY/lightgbmlib_$OS_NAME.jar
    fi
    exit 0
fi

if [[ $TASK == "lint" ]]; then
    cd ${BUILD_DIRECTORY}
    mamba create -q -y -n $CONDA_ENV \
        ${CONDA_PYTHON_REQUIREMENT} \
        'cmakelint>=1.4.2' \
        'cpplint>=1.6.0' \
        'matplotlib-base>=3.8.3' \
        'mypy>=1.8.0' \
        'pre-commit>=3.6.0' \
        'pyarrow>=6.0' \
        'r-lintr>=3.1'
    source activate $CONDA_ENV
    echo "Linting Python code"
    bash ${BUILD_DIRECTORY}/.ci/lint-python.sh || exit 1
    echo "Linting R code"
    Rscript ${BUILD_DIRECTORY}/.ci/lint_r_code.R ${BUILD_DIRECTORY} || exit 1
    echo "Linting C++ code"
    bash ${BUILD_DIRECTORY}/.ci/lint-cpp.sh || exit 1
    exit 0
fi

if [[ $TASK == "check-docs" ]] || [[ $TASK == "check-links" ]]; then
    cd $BUILD_DIRECTORY/docs
    mamba env create \
        -n $CONDA_ENV \
        --file ./env.yml || exit 1
    mamba install \
        -q \
        -y \
        -n $CONDA_ENV \
            'doxygen>=1.10.0' \
            'rstcheck>=6.2.0' || exit 1
    source activate $CONDA_ENV
    # check reStructuredText formatting
    cd $BUILD_DIRECTORY/python-package
    rstcheck --report-level warning $(find . -type f -name "*.rst") || exit 1
    cd $BUILD_DIRECTORY/docs
    rstcheck --report-level warning --ignore-directives=autoclass,autofunction,autosummary,doxygenfile $(find . -type f -name "*.rst") || exit 1
    # build docs
    make html || exit 1
    if [[ $TASK == "check-links" ]]; then
        # check docs for broken links
        pip install --user linkchecker
        linkchecker --config=.linkcheckerrc ./_build/html/*.html || exit 1
        exit 0
    fi
    # check the consistency of parameters' descriptions and other stuff
    cp $BUILD_DIRECTORY/docs/Parameters.rst $BUILD_DIRECTORY/docs/Parameters-backup.rst
    cp $BUILD_DIRECTORY/src/io/config_auto.cpp $BUILD_DIRECTORY/src/io/config_auto-backup.cpp
    python $BUILD_DIRECTORY/helpers/parameter_generator.py || exit 1
    diff $BUILD_DIRECTORY/docs/Parameters-backup.rst $BUILD_DIRECTORY/docs/Parameters.rst || exit 1
    diff $BUILD_DIRECTORY/src/io/config_auto-backup.cpp $BUILD_DIRECTORY/src/io/config_auto.cpp || exit 1
    exit 0
fi

if [[ $PYTHON_VERSION == "3.7" ]]; then
    CONDA_REQUIREMENT_FILES="--file ${BUILD_DIRECTORY}/.ci/conda-envs/ci-core-py37.txt"
else
    CONDA_REQUIREMENT_FILES="--file ${BUILD_DIRECTORY}/.ci/conda-envs/ci-core.txt"
fi

mamba create \
    -y \
    -n $CONDA_ENV \
    ${CONDA_REQUIREMENT_FILES} \
    ${CONDA_PYTHON_REQUIREMENT} \
|| exit 1

source activate $CONDA_ENV

cd $BUILD_DIRECTORY

if [[ $OS_NAME == "macos" ]] && [[ $COMPILER == "clang" ]]; then
    # fix "OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized." (OpenMP library conflict due to conda's MKL)
    for LIBOMP_ALIAS in libgomp.dylib libiomp5.dylib libomp.dylib; do sudo ln -sf "$(brew --cellar libomp)"/*/lib/libomp.dylib $CONDA_PREFIX/lib/$LIBOMP_ALIAS || exit 1; done
fi

if [[ $TASK == "sdist" ]]; then
    cd $BUILD_DIRECTORY && sh ./build-python.sh sdist || exit 1
    sh $BUILD_DIRECTORY/.ci/check_python_dists.sh $BUILD_DIRECTORY/dist || exit 1
    pip install --user $BUILD_DIRECTORY/dist/lightgbm-$LGB_VER.tar.gz -v || exit 1
    if [[ $PRODUCES_ARTIFACTS == "true" ]]; then
        cp $BUILD_DIRECTORY/dist/lightgbm-$LGB_VER.tar.gz $BUILD_ARTIFACTSTAGINGDIRECTORY || exit 1
    fi
    pytest $BUILD_DIRECTORY/tests/python_package_test || exit 1
    exit 0
elif [[ $TASK == "bdist" ]]; then
    if [[ $OS_NAME == "macos" ]]; then
        cd $BUILD_DIRECTORY && sh ./build-python.sh bdist_wheel || exit 1
        sh $BUILD_DIRECTORY/.ci/check_python_dists.sh $BUILD_DIRECTORY/dist || exit 1
        mv \
            ./dist/*.whl \
            ./dist/tmp.whl || exit 1
        if [[ $ARCH == "x86_64" ]]; then
            PLATFORM="macosx_10_15_x86_64.macosx_11_6_x86_64.macosx_12_5_x86_64"
        else
            echo "ERROR: macos wheels not supported yet on architecture '${ARCH}'"
            exit 1
        fi
        mv \
            ./dist/tmp.whl \
            dist/lightgbm-$LGB_VER-py3-none-$PLATFORM.whl || exit 1
        if [[ $PRODUCES_ARTIFACTS == "true" ]]; then
            cp dist/lightgbm-$LGB_VER-py3-none-macosx*.whl $BUILD_ARTIFACTSTAGINGDIRECTORY || exit 1
        fi
    else
        if [[ $ARCH == "x86_64" ]]; then
            PLATFORM="manylinux_2_28_x86_64"
        else
            PLATFORM="manylinux2014_$ARCH"
        fi
        cd $BUILD_DIRECTORY && sh ./build-python.sh bdist_wheel --integrated-opencl || exit 1
        mv \
            ./dist/*.whl \
            ./dist/tmp.whl || exit 1
        mv \
            ./dist/tmp.whl \
            ./dist/lightgbm-$LGB_VER-py3-none-$PLATFORM.whl || exit 1
        sh $BUILD_DIRECTORY/.ci/check_python_dists.sh $BUILD_DIRECTORY/dist || exit 1
        if [[ $PRODUCES_ARTIFACTS == "true" ]]; then
            cp dist/lightgbm-$LGB_VER-py3-none-$PLATFORM.whl $BUILD_ARTIFACTSTAGINGDIRECTORY || exit 1
        fi
        # Make sure we can do both CPU and GPU; see tests/python_package_test/test_dual.py
        export LIGHTGBM_TEST_DUAL_CPU_GPU=1
    fi
    pip install --user $BUILD_DIRECTORY/dist/*.whl || exit 1
    pytest $BUILD_DIRECTORY/tests || exit 1
    exit 0
fi

if [[ $TASK == "gpu" ]]; then
    sed -i'.bak' 's/std::string device_type = "cpu";/std::string device_type = "gpu";/' $BUILD_DIRECTORY/include/LightGBM/config.h
    grep -q 'std::string device_type = "gpu"' $BUILD_DIRECTORY/include/LightGBM/config.h || exit 1  # make sure that changes were really done
    if [[ $METHOD == "pip" ]]; then
        cd $BUILD_DIRECTORY && sh ./build-python.sh sdist || exit 1
        sh $BUILD_DIRECTORY/.ci/check_python_dists.sh $BUILD_DIRECTORY/dist || exit 1
        pip install \
            --user \
            -v \
            --config-settings=cmake.define.USE_GPU=ON \
            $BUILD_DIRECTORY/dist/lightgbm-$LGB_VER.tar.gz \
        || exit 1
        pytest $BUILD_DIRECTORY/tests/python_package_test || exit 1
        exit 0
    elif [[ $METHOD == "wheel" ]]; then
        cd $BUILD_DIRECTORY && sh ./build-python.sh bdist_wheel --gpu || exit 1
        sh $BUILD_DIRECTORY/.ci/check_python_dists.sh $BUILD_DIRECTORY/dist || exit 1
        pip install --user $BUILD_DIRECTORY/dist/lightgbm-$LGB_VER*.whl -v || exit 1
        pytest $BUILD_DIRECTORY/tests || exit 1
        exit 0
    elif [[ $METHOD == "source" ]]; then
        cmake -B build -S . -DUSE_GPU=ON
    fi
elif [[ $TASK == "cuda" ]]; then
    sed -i'.bak' 's/std::string device_type = "cpu";/std::string device_type = "cuda";/' $BUILD_DIRECTORY/include/LightGBM/config.h
    grep -q 'std::string device_type = "cuda"' $BUILD_DIRECTORY/include/LightGBM/config.h || exit 1  # make sure that changes were really done
    # by default ``gpu_use_dp=false`` for efficiency. change to ``true`` here for exact results in ci tests
    sed -i'.bak' 's/gpu_use_dp = false;/gpu_use_dp = true;/' $BUILD_DIRECTORY/include/LightGBM/config.h
    grep -q 'gpu_use_dp = true' $BUILD_DIRECTORY/include/LightGBM/config.h || exit 1  # make sure that changes were really done
    if [[ $METHOD == "pip" ]]; then
        cd $BUILD_DIRECTORY && sh ./build-python.sh sdist || exit 1
        sh $BUILD_DIRECTORY/.ci/check_python_dists.sh $BUILD_DIRECTORY/dist || exit 1
        pip install \
            --user \
            -v \
            --config-settings=cmake.define.USE_CUDA=ON \
            $BUILD_DIRECTORY/dist/lightgbm-$LGB_VER.tar.gz \
        || exit 1
        pytest $BUILD_DIRECTORY/tests/python_package_test || exit 1
        exit 0
    elif [[ $METHOD == "wheel" ]]; then
        cd $BUILD_DIRECTORY && sh ./build-python.sh bdist_wheel --cuda || exit 1
        sh $BUILD_DIRECTORY/.ci/check_python_dists.sh $BUILD_DIRECTORY/dist || exit 1
        pip install --user $BUILD_DIRECTORY/dist/lightgbm-$LGB_VER*.whl -v || exit 1
        pytest $BUILD_DIRECTORY/tests || exit 1
        exit 0
    elif [[ $METHOD == "source" ]]; then
        cmake -B build -S . -DUSE_CUDA=ON
    fi
elif [[ $TASK == "mpi" ]]; then
    if [[ $METHOD == "pip" ]]; then
        cd $BUILD_DIRECTORY && sh ./build-python.sh sdist || exit 1
        sh $BUILD_DIRECTORY/.ci/check_python_dists.sh $BUILD_DIRECTORY/dist || exit 1
        pip install \
            --user \
            -v \
            --config-settings=cmake.define.USE_MPI=ON \
            $BUILD_DIRECTORY/dist/lightgbm-$LGB_VER.tar.gz \
        || exit 1
        pytest $BUILD_DIRECTORY/tests/python_package_test || exit 1
        exit 0
    elif [[ $METHOD == "wheel" ]]; then
        cd $BUILD_DIRECTORY && sh ./build-python.sh bdist_wheel --mpi || exit 1
        sh $BUILD_DIRECTORY/.ci/check_python_dists.sh $BUILD_DIRECTORY/dist || exit 1
        pip install --user $BUILD_DIRECTORY/dist/lightgbm-$LGB_VER*.whl -v || exit 1
        pytest $BUILD_DIRECTORY/tests || exit 1
        exit 0
    elif [[ $METHOD == "source" ]]; then
        cmake -B build -S . -DUSE_MPI=ON -DUSE_DEBUG=ON
    fi
else
    cmake -B build -S .
fi

cmake --build build --target _lightgbm -j4 || exit 1

cd $BUILD_DIRECTORY && sh ./build-python.sh install --precompile --user || exit 1
pytest $BUILD_DIRECTORY/tests || exit 1

if [[ $TASK == "regular" ]]; then
    if [[ $PRODUCES_ARTIFACTS == "true" ]]; then
        if [[ $OS_NAME == "macos" ]]; then
            cp $BUILD_DIRECTORY/lib_lightgbm.dylib $BUILD_ARTIFACTSTAGINGDIRECTORY/lib_lightgbm.dylib
        else
            if [[ $COMPILER == "gcc" ]]; then
                objdump -T $BUILD_DIRECTORY/lib_lightgbm.so > $BUILD_DIRECTORY/objdump.log || exit 1
                python $BUILD_DIRECTORY/helpers/check_dynamic_dependencies.py $BUILD_DIRECTORY/objdump.log || exit 1
            fi
            cp $BUILD_DIRECTORY/lib_lightgbm.so $BUILD_ARTIFACTSTAGINGDIRECTORY/lib_lightgbm.so
        fi
    fi
    cd $BUILD_DIRECTORY/examples/python-guide
    sed -i'.bak' '/import lightgbm as lgb/a\
import matplotlib\
matplotlib.use\(\"Agg\"\)\
' plot_example.py  # prevent interactive window mode
    sed -i'.bak' 's/graph.render(view=True)/graph.render(view=False)/' plot_example.py
    # requirements for examples
    mamba install -y -n $CONDA_ENV \
        'h5py>=3.10' \
        'ipywidgets>=8.1.2' \
        'notebook>=7.1.2'
    for f in *.py **/*.py; do python $f || exit 1; done  # run all examples
    cd $BUILD_DIRECTORY/examples/python-guide/notebooks
    sed -i'.bak' 's/INTERACTIVE = False/assert False, \\"Interactive mode disabled\\"/' interactive_plot_example.ipynb
    jupyter nbconvert --ExecutePreprocessor.timeout=180 --to notebook --execute --inplace *.ipynb || exit 1  # run all notebooks

    # importing the library should succeed even if all optional dependencies are not present
    conda uninstall -n $CONDA_ENV --force --yes \
        cffi \
        dask \
        distributed \
        joblib \
        matplotlib-base \
        psutil \
        pyarrow \
        python-graphviz \
        scikit-learn || exit 1
    python -c "import lightgbm" || exit 1
fi
