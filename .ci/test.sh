#!/bin/bash

if [[ $OS_NAME == "macos" ]] && [[ $COMPILER == "gcc" ]]; then
    export CXX=g++-9
    export CC=gcc-9
elif [[ $OS_NAME == "linux" ]] && [[ $COMPILER == "clang" ]]; then
    export CXX=clang++
    export CC=clang
fi

if [[ $OS_NAME == "macos" ]] && [[ $COMPILER == "clang" ]] && [[ $(sw_vers -productVersion | cut -d'.' -f2) -ge "14" ]]; then
    CMAKE_OPTS=(-DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include"
                -DOpenMP_C_LIB_NAMES=omp
                -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include"
                -DOpenMP_CXX_LIB_NAMES=omp
                -DOpenMP_omp_LIBRARY=$(brew --prefix libomp)/lib/libomp.dylib)
else
    CMAKE_OPTS=()
fi

conda create -q -y -n $CONDA_ENV python=$PYTHON_VERSION
source activate $CONDA_ENV

cd $BUILD_DIRECTORY

if [[ $TRAVIS == "true" ]] && [[ $TASK == "check-docs" ]]; then
    cd $BUILD_DIRECTORY/docs
    conda install -q -y -n $CONDA_ENV -c conda-forge doxygen
    pip install --user -r requirements.txt rstcheck
    # check reStructuredText formatting
    cd $BUILD_DIRECTORY/python-package
    rstcheck --report warning `find . -type f -name "*.rst"` || exit -1
    cd $BUILD_DIRECTORY/docs
    rstcheck --report warning --ignore-directives=autoclass,autofunction,doxygenfile `find . -type f -name "*.rst"` || exit -1
    # build docs and check them for broken links
    make html || exit -1
    find ./_build/html/ -type f -name '*.html' -exec \
    sed -i'.bak' -e 's;\(\.\/[^.]*\.\)rst\([^[:space:]]*\);\1html\2;g' {} \;  # emulate js function
    if [[ $OS_NAME == "linux" ]]; then
        sudo apt-get update
        sudo apt-get install linkchecker
        linkchecker --config=.linkcheckerrc ./_build/html/*.html || exit -1
    fi
    # check the consistency of parameters' descriptions and other stuff
    cp $BUILD_DIRECTORY/docs/Parameters.rst $BUILD_DIRECTORY/docs/Parameters-backup.rst
    cp $BUILD_DIRECTORY/src/io/config_auto.cpp $BUILD_DIRECTORY/src/io/config_auto-backup.cpp
    python $BUILD_DIRECTORY/helpers/parameter_generator.py || exit -1
    diff $BUILD_DIRECTORY/docs/Parameters-backup.rst $BUILD_DIRECTORY/docs/Parameters.rst || exit -1
    diff $BUILD_DIRECTORY/src/io/config_auto-backup.cpp $BUILD_DIRECTORY/src/io/config_auto.cpp || exit -1
    exit 0
fi

if [[ $TASK == "lint" ]]; then
    conda install -q -y -n $CONDA_ENV pycodestyle pydocstyle
    pip install --user cpplint
    pycodestyle --ignore=E501,W503 --exclude=./compute,./.nuget . || exit -1
    pydocstyle --convention=numpy --add-ignore=D105 --match-dir="^(?!^compute|test|example).*" --match="(?!^test_|setup).*\.py" . || exit -1
    cpplint --filter=-build/c++11,-build/include_subdir,-build/header_guard,-whitespace/line_length --recursive ./src ./include || exit 0
    exit 0
fi

if [[ $TASK == "if-else" ]]; then
    conda install -q -y -n $CONDA_ENV numpy
    mkdir $BUILD_DIRECTORY/build && cd $BUILD_DIRECTORY/build && cmake "${CMAKE_OPTS[@]}" .. && make lightgbm -j4 || exit -1
    cd $BUILD_DIRECTORY/tests/cpp_test && ../../lightgbm config=train.conf convert_model_language=cpp convert_model=../../src/boosting/gbdt_prediction.cpp && ../../lightgbm config=predict.conf output_result=origin.pred || exit -1
    cd $BUILD_DIRECTORY/build && make lightgbm -j4 || exit -1
    cd $BUILD_DIRECTORY/tests/cpp_test && ../../lightgbm config=predict.conf output_result=ifelse.pred && python test.py || exit -1
    exit 0
fi

conda install -q -y -n $CONDA_ENV joblib matplotlib numpy pandas psutil pytest python-graphviz scikit-learn scipy

if [[ $OS_NAME == "macos" ]] && [[ $COMPILER == "clang" ]]; then
    # fix "OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized." (OpenMP library conflict due to conda's MKL)
    for LIBOMP_ALIAS in libgomp.dylib libiomp5.dylib libomp.dylib; do sudo ln -sf "$(brew --cellar libomp)"/*/lib/libomp.dylib $CONDA_PREFIX/lib/$LIBOMP_ALIAS || exit -1; done
fi

if [[ $TASK == "sdist" ]]; then
    cd $BUILD_DIRECTORY/python-package && python setup.py sdist || exit -1
    pip install --user $BUILD_DIRECTORY/python-package/dist/lightgbm-$LGB_VER.tar.gz -v || exit -1
    if [[ $AZURE == "true" ]]; then
        cp $BUILD_DIRECTORY/python-package/dist/lightgbm-$LGB_VER.tar.gz $BUILD_ARTIFACTSTAGINGDIRECTORY
        mkdir $BUILD_DIRECTORY/build && cd $BUILD_DIRECTORY/build
        if [[ $OS_NAME == "macos" ]]; then
            cmake -DUSE_SWIG=ON -DAPPLE_OUTPUT_DYLIB=ON "${CMAKE_OPTS[@]}" ..
        else
            cmake -DUSE_SWIG=ON "${CMAKE_OPTS[@]}" ..
        fi
        make -j4 || exit -1
        if [[ $OS_NAME == "linux" ]] && [[ $COMPILER == "gcc" ]]; then
            objdump -T $BUILD_DIRECTORY/lib_lightgbm.so > $BUILD_DIRECTORY/objdump.log || exit -1
            objdump -T $BUILD_DIRECTORY/lib_lightgbm_swig.so >> $BUILD_DIRECTORY/objdump.log || exit -1
            python $BUILD_DIRECTORY/helpers/check_dynamic_dependencies.py $BUILD_DIRECTORY/objdump.log || exit -1
        fi
        cp $BUILD_DIRECTORY/build/lightgbmlib.jar $BUILD_ARTIFACTSTAGINGDIRECTORY/lightgbmlib_$OS_NAME.jar
    fi
    pytest $BUILD_DIRECTORY/tests/python_package_test || exit -1
    exit 0
elif [[ $TASK == "bdist" ]]; then
    if [[ $OS_NAME == "macos" ]]; then
        cd $BUILD_DIRECTORY/python-package && python setup.py bdist_wheel --plat-name=macosx --universal || exit -1
        mv dist/lightgbm-$LGB_VER-py2.py3-none-macosx.whl dist/lightgbm-$LGB_VER-py2.py3-none-macosx_10_8_x86_64.macosx_10_9_x86_64.macosx_10_10_x86_64.macosx_10_11_x86_64.macosx_10_12_x86_64.macosx_10_13_x86_64.macosx_10_14_x86_64.whl
        if [[ $AZURE == "true" ]]; then
            cp dist/lightgbm-$LGB_VER-py2.py3-none-macosx*.whl $BUILD_ARTIFACTSTAGINGDIRECTORY
        fi
    else
        cd $BUILD_DIRECTORY/python-package && python setup.py bdist_wheel --plat-name=manylinux1_x86_64 --universal || exit -1
        if [[ $AZURE == "true" ]]; then
            cp dist/lightgbm-$LGB_VER-py2.py3-none-manylinux1_x86_64.whl $BUILD_ARTIFACTSTAGINGDIRECTORY
        fi
    fi
    pip install --user $BUILD_DIRECTORY/python-package/dist/*.whl || exit -1
    pytest $BUILD_DIRECTORY/tests || exit -1
    exit 0
fi

mkdir $BUILD_DIRECTORY/build && cd $BUILD_DIRECTORY/build

if [[ $TASK == "gpu" ]]; then
    sed -i'.bak' 's/std::string device_type = "cpu";/std::string device_type = "gpu";/' $BUILD_DIRECTORY/include/LightGBM/config.h
    grep -q 'std::string device_type = "gpu"' $BUILD_DIRECTORY/include/LightGBM/config.h || exit -1  # make sure that changes were really done
    if [[ $METHOD == "pip" ]]; then
        cd $BUILD_DIRECTORY/python-package && python setup.py sdist || exit -1
        pip install --user $BUILD_DIRECTORY/python-package/dist/lightgbm-$LGB_VER.tar.gz -v --install-option=--gpu --install-option="--opencl-include-dir=$AMDAPPSDK_PATH/include/" || exit -1
        pytest $BUILD_DIRECTORY/tests/python_package_test || exit -1
        exit 0
    fi
    cmake -DUSE_GPU=ON -DOpenCL_INCLUDE_DIR=$AMDAPPSDK_PATH/include/ "${CMAKE_OPTS[@]}" ..
elif [[ $TASK == "mpi" ]]; then
    if [[ $METHOD == "pip" ]]; then
        cd $BUILD_DIRECTORY/python-package && python setup.py sdist || exit -1
        pip install --user $BUILD_DIRECTORY/python-package/dist/lightgbm-$LGB_VER.tar.gz -v --install-option=--mpi || exit -1
        pytest $BUILD_DIRECTORY/tests/python_package_test || exit -1
        exit 0
    fi
    cmake -DUSE_MPI=ON "${CMAKE_OPTS[@]}" ..
else
    cmake "${CMAKE_OPTS[@]}" ..
fi

make _lightgbm -j4 || exit -1

cd $BUILD_DIRECTORY/python-package && python setup.py install --precompile --user || exit -1
pytest $BUILD_DIRECTORY/tests || exit -1

if [[ $TASK == "regular" ]]; then
    if [[ $AZURE == "true" ]]; then
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
    for f in *.py; do python $f || exit -1; done  # run all examples
    cd $BUILD_DIRECTORY/examples/python-guide/notebooks
    conda install -q -y -n $CONDA_ENV ipywidgets notebook
    jupyter nbconvert --ExecutePreprocessor.timeout=180 --to notebook --execute --inplace *.ipynb || exit -1  # run all notebooks
fi
