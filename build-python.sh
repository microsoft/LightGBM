#!/bin/sh

set -e -u

# Default values of arguments
INSTALL="false"
BUILD_SDIST="false"
BUILD_WHEEL="false"

PIP_INSTALL_ARGS=""
BUILD_ARGS=""
PRECOMPILE="false"

BOOST_INCLUDE_DIR=""
BOOST_LIBRARY_DIR=""
BOOST_ROOT=""
OPENCL_INCLUDE_DIR=""
OPENCL_LIBRARY=""

while [ $# -gt 0 ]; do
  case "$1" in
    ############################
    # sub-commands of setup.py #
    ############################
    install)
      INSTALL="true"
      ;;
    sdist)
      BUILD_SDIST="true"
      ;;
    bdist_wheel)
      BUILD_WHEEL="true"
      ;;
    ############################
    # customized library paths #
    ############################
    --boost-include-dir|--boost-include-dir=*)
        if [[ "$1" != *=* ]];
            then shift;
        fi
        BOOST_INCLUDE_DIR="${1#*=}"
        ;;
    --boost-librarydir|--boost-librarydir=*)
        if [[ "$1" != *=* ]];
            then shift;
        fi
        BOOST_LIBRARY_DIR="${1#*=}"
        ;;
    --boost-root|--boost-root=*)
        if [[ "$1" != *=* ]];
            then shift;
        fi
        BOOST_ROOT="${1#*=}"
        ;;
    --opencl-include-dir|--opencl-include-dir=*)
        if [[ "$1" != *=* ]];
            then shift;
        fi
        OPENCL_INCLUDE_DIR="${1#*=}"
        ;;
    --opencl-library|--opencl-library=*)
        if [[ "$1" != *=* ]];
            then shift;
        fi
        OPENCL_LIBRARY="${1#*=}"
        ;;
    #########
    # flags #
    #########
    --bit32)
        BUILD_ARGS="${BUILD_ARGS} --bit32"
        ;;
    --cuda)
        BUILD_ARGS="${BUILD_ARGS} --cuda"
        ;;
    --gpu)
        BUILD_ARGS="${BUILD_ARGS} --gpu"
        ;;
    --hdfs)
        BUILD_ARGS="${BUILD_ARGS} --hdfs"
        ;;
    --integrated-opencl)
        BUILD_ARGS="${BUILD_ARGS} --integrated-opencl"
        ;;
    --mingw)
        BUILD_ARGS="${BUILD_ARGS} --mingw"
        ;;
    --mpi)
        BUILD_ARGS="${BUILD_ARGS} --mpi"
        ;;
    --nomp)
      BUILD_ARGS="${BUILD_ARGS} --nomp"
      ;;
    --precompile)
      PRECOMPILE="true"
      ;;
    --time-costs)
      BUILD_ARGS="${PIP_INSTALL_ARGS} --time-costs"
      ;;
    --user)
      PIP_INSTALL_ARGS="${PIP_INSTALL_ARGS} --user"
      ;;
    *)
      echo "invalid argument '${1}'"
      exit -1
      ;;
  esac
  shift
done

# create a new directory that just contains the files needed
# to build the Python package
create_isolated_source_dir() {
    rm -rf \
        ./lightgbm-python \
        ./lightgbm \
        ./python-package/build \
        ./python-package/build_cpp \
        ./python-package/compile \
        ./python-package/dist \
        ./python-package/lightgbm.egg-info

    cp -R ./python-package ./lightgbm-python
    cp -R ./cmake ./lightgbm-python/
    cp CMakeLists.txt ./lightgbm-python/
    cp -R ./include ./lightgbm-python/
    cp LICENSE ./lightgbm-python/
    cp -R ./src ./lightgbm-python/
    cp -R ./swig ./lightgbm-python/
    cp VERSION.txt ./lightgbm-python/
    cp -R ./windows ./lightgbm-python/

    # include only specific files from external_libs, to keep the package
    # small and avoid redistributing code with licenses incompatible with
    # LightGBM's license
    mkdir -p ./lightgbm-python/external_libs/fast_double_parser/include/
    cp \
        external_libs/fast_double_parser/include/fast_double_parser.h \
        ./lightgbm-python/external_libs/fast_double_parser/include/

    mkdir -p ./lightgbm-python/external_libs/fmt/include/fmt
    cp \
        external_libs/fmt/include/fmt/*.h \
        ./lightgbm-python/external_libs/fmt/include/fmt/

    mkdir -p ./lightgbm-python/external_libs/eigen/Eigen

    modules="Cholesky Core Dense Eigenvalues Geometry Householder Jacobi LU QR SVD"
    for eigen_module in ${modules}; do
        cp \
            external_libs/eigen/Eigen/${eigen_module} \
            ./lightgbm-python/external_libs/eigen/Eigen/${eigen_module}
        if [ ${eigen_module} != "Dense" ]; then
            mkdir -p ./lightgbm-python/external_libs/eigen/Eigen/src/${eigen_module}/
            cp \
                -R \
                external_libs/eigen/Eigen/src/${eigen_module}/* \
                ./lightgbm-python/external_libs/eigen/Eigen/src/${eigen_module}/
        fi
    done

    mkdir -p ./lightgbm-python/external_libs/eigen/Eigen/misc
    cp \
        -R \
        external_libs/eigen/Eigen/src/misc \
        ./lightgbm-python/external_libs/eigen/Eigen/src/misc/

    mkdir -p ./lightgbm-python/external_libs/eigen/Eigen/plugins
    cp \
        -R \
        external_libs/eigen/Eigen/src/plugins \
        ./lightgbm-python/external_libs/eigen/Eigen/src/plugins/

    mkdir -p ./lightgbm-python/external_libs/compute
    cp \
        -R \
        external_libs/compute/cmake \
        ./lightgbm-python/external_libs/cmake/
    cp \
        -R \
        external_libs/compute/include \
        ./lightgbm-python/external_libs/include/
    cp \
        -R \
        external_libs/compute/meta \
        ./lightgbm-python/external_libs/meta/
}

create_isolated_source_dir

cd ./lightgbm-python

# installation involves building the wheel + `pip install`-ing it
if test "${INSTALL}" = true; then
    if test "${PRECOMPILE}" = true; then
        BUILD_SDIST=true
        BUILD_WHEEL=false
        BUILD_ARGS=""
        rm -rf \
            ./cmake \
            ./CMakeLists.txt \
            ./external_libs \
            ./include \
            ./src \
            ./swig \
            ./windows
        if test -f ../lib_lightgbm.so; then
            cp ../lib_lightgbm.so ./lightgbm/lib_lightgbm.so
        fi
        if test -f ../Release/lib_lightgbm.dll; then
            cp ../Release/lib_lightgbm.dll ./lightgbm/lib_lightgbm.dll
        fi
        rm -f ./*.bak
    else
        BUILD_WHEEL=true
    fi
fi

if test "${BUILD_SDIST}" = true; then
    echo "--- building sdist ---"
    rm -f ../dist/*.tar.gz
    python ./setup.py sdist \
        --dist-dir ../dist
fi

if test "${BUILD_WHEEL}" = true; then
    echo "--- building wheel ---"
    rm -f ../dist/*.whl
    python setup.py bdist_wheel \
        --dist-dir ../dist \
        ${BUILD_ARGS}
fi

if test "${INSTALL}" = true; then
    if test "${PRECOMPILE}" = true; then
        pip install ${PIP_INSTALL_ARGS} ../dist/*.tar.gz
    else
        pip install ${PIP_INSTALL_ARGS} ../dist/*.whl
    fi
fi

echo "cleaning up"
rm -rf ./lightgbm-python
