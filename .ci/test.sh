#!/bin/bash

if [[ $OS_NAME == "macos" ]] && [[ $COMPILER == "gcc" ]]; then
    export CXX=g++-11
    export CC=gcc-11
elif [[ $OS_NAME == "linux" ]] && [[ $COMPILER == "clang" ]]; then
    export CXX=clang++
    export CC=clang
fi

if [[ $IN_UBUNTU_BASE_CONTAINER == "true" ]]; then
    export LANG="en_US.UTF-8"
    export LC_ALL="en_US.UTF-8"
fi

if [[ "${TASK}" == "r-package" ]] || [[ "${TASK}" == "r-rchk" ]]; then
    bash ${BUILD_DIRECTORY}/.ci/test_r_package.sh || exit -1
    exit 0
fi

if [[ "$TASK" == "cpp-tests" ]]; then
    mkdir $BUILD_DIRECTORY/build && cd $BUILD_DIRECTORY/build
    if [[ $METHOD == "with-sanitizers" ]]; then
        extra_cmake_opts="-DUSE_SANITIZER=ON"
        if [[ -n $SANITIZERS ]]; then
            extra_cmake_opts="$extra_cmake_opts -DENABLED_SANITIZERS=$SANITIZERS"
        fi
    else
        extra_cmake_opts=""
    fi
    cmake -DBUILD_CPP_TEST=ON -DUSE_OPENMP=OFF -DUSE_DEBUG=ON $extra_cmake_opts ..
    make testlightgbm -j4 || exit -1
    ./../testlightgbm || exit -1
    exit 0
fi

CONDA_PYTHON_REQUIREMENT="python=$PYTHON_VERSION[build=*cpython]"

if [[ $TASK == "if-else" ]]; then
    conda create -q -y -n $CONDA_ENV ${CONDA_PYTHON_REQUIREMENT} numpy
    source activate $CONDA_ENV
    mkdir $BUILD_DIRECTORY/build && cd $BUILD_DIRECTORY/build && cmake .. && make lightgbm -j4 || exit -1
    cd $BUILD_DIRECTORY/tests/cpp_tests && ../../lightgbm config=train.conf convert_model_language=cpp convert_model=../../src/boosting/gbdt_prediction.cpp && ../../lightgbm config=predict.conf output_result=origin.pred || exit -1
    cd $BUILD_DIRECTORY/build && make lightgbm -j4 || exit -1
    cd $BUILD_DIRECTORY/tests/cpp_tests && ../../lightgbm config=predict.conf output_result=ifelse.pred && python test.py || exit -1
    exit 0
fi

if [[ $TASK == "swig" ]]; then
    mkdir $BUILD_DIRECTORY/build && cd $BUILD_DIRECTORY/build
    if [[ $OS_NAME == "macos" ]]; then
        cmake -DUSE_SWIG=ON -DAPPLE_OUTPUT_DYLIB=ON ..
    else
        cmake -DUSE_SWIG=ON ..
    fi
    make -j4 || exit -1
    if [[ $OS_NAME == "linux" ]] && [[ $COMPILER == "gcc" ]]; then
        objdump -T $BUILD_DIRECTORY/lib_lightgbm.so > $BUILD_DIRECTORY/objdump.log || exit -1
        objdump -T $BUILD_DIRECTORY/lib_lightgbm_swig.so >> $BUILD_DIRECTORY/objdump.log || exit -1
        python $BUILD_DIRECTORY/helpers/check_dynamic_dependencies.py $BUILD_DIRECTORY/objdump.log || exit -1
    fi
    if [[ $PRODUCES_ARTIFACTS == "true" ]]; then
        cp $BUILD_DIRECTORY/build/lightgbmlib.jar $BUILD_ARTIFACTSTAGINGDIRECTORY/lightgbmlib_$OS_NAME.jar
    fi
    exit 0
fi

if [[ $TASK == "lint" ]]; then
    cd ${BUILD_DIRECTORY}
    conda create -q -y -n $CONDA_ENV \
        ${CONDA_PYTHON_REQUIREMENT} \
        cmakelint \
        cpplint \
        isort \
        mypy \
        'r-lintr>=3.0' \
        ruff
    source activate $CONDA_ENV
    echo "Linting Python code"
    sh ${BUILD_DIRECTORY}/.ci/lint-python.sh || exit -1
    echo "Linting R code"
    Rscript ${BUILD_DIRECTORY}/.ci/lint_r_code.R ${BUILD_DIRECTORY} || exit -1
    echo "Linting C++ code"
    sh ${BUILD_DIRECTORY}/.ci/lint-cpp.sh || exit -1
    exit 0
fi

if [[ $TASK == "check-docs" ]] || [[ $TASK == "check-links" ]]; then
    cd $BUILD_DIRECTORY/docs
    conda env create \
        -n $CONDA_ENV \
        --file ./env.yml || exit -1
    conda install \
        -q \
        -y \
        -n $CONDA_ENV \
            doxygen \
            'rstcheck>=6.0.0' || exit -1
    source activate $CONDA_ENV
    # check reStructuredText formatting
    cd $BUILD_DIRECTORY/python-package
    rstcheck --report-level warning $(find . -type f -name "*.rst") || exit -1
    cd $BUILD_DIRECTORY/docs
    rstcheck --report-level warning --ignore-directives=autoclass,autofunction,autosummary,doxygenfile $(find . -type f -name "*.rst") || exit -1
    # build docs
    make html || exit -1
    if [[ $TASK == "check-links" ]]; then
        # check docs for broken links
        pip install --user linkchecker
        linkchecker --config=.linkcheckerrc ./_build/html/*.html || exit -1
        exit 0
    fi
    # check the consistency of parameters' descriptions and other stuff
    cp $BUILD_DIRECTORY/docs/Parameters.rst $BUILD_DIRECTORY/docs/Parameters-backup.rst
    cp $BUILD_DIRECTORY/src/io/config_auto.cpp $BUILD_DIRECTORY/src/io/config_auto-backup.cpp
    python $BUILD_DIRECTORY/helpers/parameter_generator.py || exit -1
    diff $BUILD_DIRECTORY/docs/Parameters-backup.rst $BUILD_DIRECTORY/docs/Parameters.rst || exit -1
    diff $BUILD_DIRECTORY/src/io/config_auto-backup.cpp $BUILD_DIRECTORY/src/io/config_auto.cpp || exit -1
    exit 0
fi

# including python=version[build=*cpython] to ensure that conda doesn't fall back to pypy
conda create -q -y -n $CONDA_ENV \
    cloudpickle \
    dask-core \
    distributed \
    joblib \
    matplotlib \
    numpy \
    pandas \
    psutil \
    pytest \
    ${CONDA_PYTHON_REQUIREMENT} \
    python-graphviz \
    scikit-learn \
    scipy || exit -1

source activate $CONDA_ENV

cd $BUILD_DIRECTORY

if [[ $OS_NAME == "macos" ]] && [[ $COMPILER == "clang" ]]; then
    # fix "OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized." (OpenMP library conflict due to conda's MKL)
    for LIBOMP_ALIAS in libgomp.dylib libiomp5.dylib libomp.dylib; do sudo ln -sf "$(brew --cellar libomp)"/*/lib/libomp.dylib $CONDA_PREFIX/lib/$LIBOMP_ALIAS || exit -1; done
fi

if [[ $TASK == "sdist" ]]; then
    cd $BUILD_DIRECTORY && sh ./build-python.sh sdist || exit -1
    sh $BUILD_DIRECTORY/.ci/check_python_dists.sh $BUILD_DIRECTORY/dist || exit -1
    pip install --user $BUILD_DIRECTORY/dist/lightgbm-$LGB_VER.tar.gz -v || exit -1
    if [[ $PRODUCES_ARTIFACTS == "true" ]]; then
        cp $BUILD_DIRECTORY/dist/lightgbm-$LGB_VER.tar.gz $BUILD_ARTIFACTSTAGINGDIRECTORY || exit -1
    fi
    pytest $BUILD_DIRECTORY/tests/python_package_test || exit -1
    exit 0
elif [[ $TASK == "bdist" ]]; then
    if [[ $OS_NAME == "macos" ]]; then
        cd $BUILD_DIRECTORY && sh ./build-python.sh bdist_wheel || exit -1
        sh $BUILD_DIRECTORY/.ci/check_python_dists.sh $BUILD_DIRECTORY/dist || exit -1
        mv \
            ./dist/*.whl \
            dist/lightgbm-$LGB_VER-py3-none-macosx_10_15_x86_64.macosx_11_6_x86_64.macosx_12_5_x86_64.whl || exit -1
        if [[ $PRODUCES_ARTIFACTS == "true" ]]; then
            cp dist/lightgbm-$LGB_VER-py3-none-macosx*.whl $BUILD_ARTIFACTSTAGINGDIRECTORY || exit -1
        fi
    else
        ARCH=$(uname -m)
        if [[ $ARCH == "x86_64" ]]; then
            PLATFORM="manylinux_2_28_x86_64"
        else
            PLATFORM="manylinux2014_$ARCH"
        fi
        cd $BUILD_DIRECTORY && sh ./build-python.sh bdist_wheel --integrated-opencl || exit -1
        mv \
            ./dist/*.whl \
            ./dist/lightgbm-$LGB_VER-py3-none-$PLATFORM.whl || exit -1
        sh $BUILD_DIRECTORY/.ci/check_python_dists.sh $BUILD_DIRECTORY/dist || exit -1
        if [[ $PRODUCES_ARTIFACTS == "true" ]]; then
            cp dist/lightgbm-$LGB_VER-py3-none-$PLATFORM.whl $BUILD_ARTIFACTSTAGINGDIRECTORY || exit -1
        fi
        # Make sure we can do both CPU and GPU; see tests/python_package_test/test_dual.py
        export LIGHTGBM_TEST_DUAL_CPU_GPU=1
    fi
    pip install --user $BUILD_DIRECTORY/dist/*.whl || exit -1
    pytest $BUILD_DIRECTORY/tests || exit -1
    exit 0
fi

# temporarily pin pip to versions that support 'pip install --install-option'
# ref: https://github.com/microsoft/LightGBM/issues/5061#issuecomment-1510642287
if [[ $METHOD == "pip" ]]; then
    pip install 'pip<23.1'
fi

if [[ $TASK == "gpu" ]]; then
    sed -i'.bak' 's/std::string device_type = "cpu";/std::string device_type = "gpu";/' $BUILD_DIRECTORY/include/LightGBM/config.h
    grep -q 'std::string device_type = "gpu"' $BUILD_DIRECTORY/include/LightGBM/config.h || exit -1  # make sure that changes were really done
    if [[ $METHOD == "pip" ]]; then
        cd $BUILD_DIRECTORY && sh ./build-python.sh sdist || exit -1
        sh $BUILD_DIRECTORY/.ci/check_python_dists.sh $BUILD_DIRECTORY/dist || exit -1
        pip install \
            --user \
            -v \
            --install-option=--gpu \
            $BUILD_DIRECTORY/dist/lightgbm-$LGB_VER.tar.gz \
        || exit -1
        pytest $BUILD_DIRECTORY/tests/python_package_test || exit -1
        exit 0
    elif [[ $METHOD == "wheel" ]]; then
        cd $BUILD_DIRECTORY && sh ./build-python.sh bdist_wheel --gpu || exit -1
        sh $BUILD_DIRECTORY/.ci/check_python_dists.sh $BUILD_DIRECTORY/dist || exit -1
        pip install --user $BUILD_DIRECTORY/dist/lightgbm-$LGB_VER*.whl -v || exit -1
        pytest $BUILD_DIRECTORY/tests || exit -1
        exit 0
    elif [[ $METHOD == "source" ]]; then
        mkdir $BUILD_DIRECTORY/build
        cd $BUILD_DIRECTORY/build
        cmake -DUSE_GPU=ON ..
    fi
elif [[ $TASK == "cuda" ]]; then
    sed -i'.bak' 's/std::string device_type = "cpu";/std::string device_type = "cuda";/' $BUILD_DIRECTORY/include/LightGBM/config.h
    grep -q 'std::string device_type = "cuda"' $BUILD_DIRECTORY/include/LightGBM/config.h || exit -1  # make sure that changes were really done
    # by default ``gpu_use_dp=false`` for efficiency. change to ``true`` here for exact results in ci tests
    sed -i'.bak' 's/gpu_use_dp = false;/gpu_use_dp = true;/' $BUILD_DIRECTORY/include/LightGBM/config.h
    grep -q 'gpu_use_dp = true' $BUILD_DIRECTORY/include/LightGBM/config.h || exit -1  # make sure that changes were really done
    if [[ $METHOD == "pip" ]]; then
        cd $BUILD_DIRECTORY && sh ./build-python.sh sdist || exit -1
        sh $BUILD_DIRECTORY/.ci/check_python_dists.sh $BUILD_DIRECTORY/dist || exit -1
        pip install \
            --user \
            -v \
            --install-option=--cuda \
            $BUILD_DIRECTORY/dist/lightgbm-$LGB_VER.tar.gz \
        || exit -1
        pytest $BUILD_DIRECTORY/tests/python_package_test || exit -1
        exit 0
    elif [[ $METHOD == "wheel" ]]; then
        cd $BUILD_DIRECTORY && sh ./build-python.sh bdist_wheel --cuda || exit -1
        sh $BUILD_DIRECTORY/.ci/check_python_dists.sh $BUILD_DIRECTORY/dist || exit -1
        pip install --user $BUILD_DIRECTORY/dist/lightgbm-$LGB_VER*.whl -v || exit -1
        pytest $BUILD_DIRECTORY/tests || exit -1
        exit 0
    elif [[ $METHOD == "source" ]]; then
        mkdir $BUILD_DIRECTORY/build
        cd $BUILD_DIRECTORY/build
        cmake -DUSE_CUDA=ON ..
    fi
elif [[ $TASK == "mpi" ]]; then
    if [[ $METHOD == "pip" ]]; then
        cd $BUILD_DIRECTORY && sh ./build-python.sh sdist || exit -1
        sh $BUILD_DIRECTORY/.ci/check_python_dists.sh $BUILD_DIRECTORY/dist || exit -1
        pip install \
            --user \
            -v \
            --install-option=--mpi \
            $BUILD_DIRECTORY/dist/lightgbm-$LGB_VER.tar.gz \
        || exit -1
        pytest $BUILD_DIRECTORY/tests/python_package_test || exit -1
        exit 0
    elif [[ $METHOD == "wheel" ]]; then
        cd $BUILD_DIRECTORY && sh ./build-python.sh bdist_wheel --mpi || exit -1
        sh $BUILD_DIRECTORY/.ci/check_python_dists.sh $BUILD_DIRECTORY/dist || exit -1
        pip install --user $BUILD_DIRECTORY/dist/lightgbm-$LGB_VER*.whl -v || exit -1
        pytest $BUILD_DIRECTORY/tests || exit -1
        exit 0
    elif [[ $METHOD == "source" ]]; then
        mkdir $BUILD_DIRECTORY/build
        cd $BUILD_DIRECTORY/build
        cmake -DUSE_MPI=ON -DUSE_DEBUG=ON ..
    fi
else
    mkdir $BUILD_DIRECTORY/build
    cd $BUILD_DIRECTORY/build
    cmake ..
fi

make _lightgbm -j4 || exit -1

cd $BUILD_DIRECTORY && sh ./build-python.sh install --precompile --user || exit -1
pytest $BUILD_DIRECTORY/tests || exit -1

if [[ $TASK == "regular" ]]; then
    if [[ $PRODUCES_ARTIFACTS == "true" ]]; then
        if [[ $OS_NAME == "macos" ]]; then
            cp $BUILD_DIRECTORY/lib_lightgbm.so $BUILD_ARTIFACTSTAGINGDIRECTORY/lib_lightgbm.dylib
        else
            if [[ $COMPILER == "gcc" ]]; then
                objdump -T $BUILD_DIRECTORY/lib_lightgbm.so > $BUILD_DIRECTORY/objdump.log || exit -1
                python $BUILD_DIRECTORY/helpers/check_dynamic_dependencies.py $BUILD_DIRECTORY/objdump.log || exit -1
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
    conda install -q -y -n $CONDA_ENV \
        h5py \
        ipywidgets \
        notebook
    for f in *.py **/*.py; do python $f || exit -1; done  # run all examples
    cd $BUILD_DIRECTORY/examples/python-guide/notebooks
    sed -i'.bak' 's/INTERACTIVE = False/assert False, \\"Interactive mode disabled\\"/' interactive_plot_example.ipynb
    jupyter nbconvert --ExecutePreprocessor.timeout=180 --to notebook --execute --inplace *.ipynb || exit -1  # run all notebooks

    # importing the library should succeed even if all optional dependencies are not present
    conda uninstall --force --yes \
        dask \
        distributed \
        joblib \
        matplotlib \
        psutil \
        python-graphviz \
        scikit-learn || exit -1
    python -c "import lightgbm" || exit -1
fi
