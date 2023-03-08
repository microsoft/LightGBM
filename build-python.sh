#!/bin/sh

set -e -u

# Default values of arguments
INSTALL="false"
BUILD_SDIST="false"
BUILD_WHEEL="false"

PIP_INSTALL_ARGS=""
BUILD_ARGS=""
PRECOMPILE="false"

while [ $# -gt 0 ]; do
  case "$1" in
    # sub-commands of setup.py
    install)
      INSTALL="true"
      ;;
    sdist)
      BUILD_SDIST="true"
      ;;
    bdist_wheel)
      BUILD_WHEEL="true"
      ;;
    # customizations
    --cuda)
        BUILD_ARGS="${BUILD_ARGS} --config-setting=cmake.define.USE_CUDA=ON"
        ;;
    --gpu)
        BUILD_ARGS="${BUILD_ARGS} --config-setting=cmake.define.USE_GPU=ON"
        ;;
    --hdfs)
        BUILD_ARGS="${BUILD_ARGS} --config-setting=cmake.define.USE_HDFS=ON"
        ;;
    --integrated-opencl)
        BUILD_ARGS="${BUILD_ARGS} --config-setting=cmake.define.__INTEGRATE_OPENCL=ON"
        ;;
    --mingw)
        BUILD_ARGS="${BUILD_ARGS} --config-setting=cmake.args=[-G\"MinGW Makefiles\"]"
        ;;
    --mpi)
        BUILD_ARGS="${BUILD_ARGS} --config-setting=cmake.define.USE_MPI=ON"
        ;;
    --nomp)
      BUILD_ARGS="${BUILD_ARGS} --config-setting=cmake.define.USE_OPENMP=OFF"
      ;;
    --precompile)
      PRECOMPILE="true"
      ;;
    --time-costs)
      BUILD_ARGS="${BUILD_ARGS} --config-setting=cmake.define.USE_TIMETAG=ON"
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

# TODO:
# --'mingw', -m
# -- 'bit32'
# --precompile, -p
# --boost-root=
# --boost-include-dir=
# --boost-librarydir=
# --opencl-include-dir=
# --opencl-library=
pip install --prefer-binary build

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
        # use regular-old setuptools for these builds, to avoid
        # trying to recompile the shared library
        sed -i.bak -e '/start:build-system/,/end:build-system/d' pyproject.toml
        echo '[build-system]' > ./pyproject.toml
        echo 'requires = ["setuptools"]' >> ./pyproject.toml
        echo 'build-backend = "setuptools.build_meta"' >> ./pyproject.toml
        echo "" >> ./pyproject.toml
        echo "recursive-include lightgbm *.dll *.so" > ./MANIFEST.in
        echo "" >> ./MANIFEST.in
        if test -f ../lib_lightgbm.so; then
            cp ../lib_lightgbm.so ./lightgbm/lib_lightgbm.so
        fi
        ls ..
        ls ../Release
        exit 123
        if test -f ../Release/lib_lightgbm.dll; then
            cp ../Release/lib_lightgbm.dll ./lightgbm/lib_lightgbm.dll
        fi
    else
        BUILD_WHEEL=true
    fi
fi

if test "${BUILD_SDIST}" = true; then
    echo "--- building sdist ---"
    rm -f ../dist/*.tar.gz
    python -m build \
        --sdist \
        --outdir ../dist \
        .
fi

if test "${BUILD_WHEEL}" = true; then
    echo "--- building wheel ---"
    rm -f ../dist/*.whl
    MAKEFLAGS="-j3" \
    python -m build \
        --wheel \
        --outdir ../dist \
        ${BUILD_ARGS} \
        .
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
