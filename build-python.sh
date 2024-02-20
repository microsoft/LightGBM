#!/bin/sh

# [description]
#
#     Prepare a source distribution (sdist) or built distribution (wheel)
#     of the Python package, and optionally install it.
#
# [usage]
#
#     # build sdist and put it in dist/
#     sh ./build-python.sh sdist
#
#     # build wheel and put it in dist/
#     sh ./build-python.sh bdist_wheel [OPTIONS]
#
#     # compile lib_lightgbm and install the Python package wrapping it
#     sh ./build-python.sh install [OPTIONS]
#
#     # install the Python package using a pre-compiled lib_lightgbm
#     # (assumes lib_lightgbm.{dll,so} is located at the root of the repo)
#     sh ./build-python.sh install --precompile
#
# [options]
#
#     --boost-dir=FILEPATH
#                                   Directory with Boost package configuration file.
#     --boost-include-dir=FILEPATH
#                                   Directory containing Boost headers.
#     --boost-librarydir=FILEPATH
#                                   Preferred Boost library directory.
#     --boost-root=FILEPATH
#                                   Boost preferred installation prefix.
#     --opencl-include-dir=FILEPATH
#                                   OpenCL include directory.
#     --opencl-library=FILEPATH
#                                   Path to OpenCL library.
#     --bit32
#                                   Compile 32-bit version.
#     --cuda
#                                   Compile CUDA version.
#     --gpu
#                                   Compile GPU version.
#     --hdfs
#                                   Compile HDFS version.
#     --integrated-opencl
#                                   Compile integrated OpenCL version.
#     --mingw
#                                   Compile with MinGW.
#     --mpi
#                                   Compile MPI version.
#     --no-isolation
#                                   Assume all build and install dependencies are already installed,
#                                   don't go to the internet to get them.
#     --nomp
#                                   Compile version without OpenMP support.
#     --precompile
#                                   Use precompiled library.
#                                   Only used with 'install' command.
#     --time-costs
#                                   Output time costs for different internal routines.
#     --user
#                                   Install into user-specific instead of global site-packages directory.
#                                   Only used with 'install' command.

set -e -u

echo "building lightgbm"

# Default values of arguments
INSTALL="false"
BUILD_SDIST="false"
BUILD_WHEEL="false"

PIP_INSTALL_ARGS=""
BUILD_ARGS=""
PRECOMPILE="false"

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
    --boost-dir|--boost-dir=*)
        if [[ "$1" != *=* ]];
            then shift;
        fi
        BOOST_DIR="${1#*=}"
        BUILD_ARGS="${BUILD_ARGS} --config-setting=cmake.define.Boost_DIR='${BOOST_DIR}'"
        ;;
    --boost-include-dir|--boost-include-dir=*)
        if [[ "$1" != *=* ]];
            then shift;
        fi
        BOOST_INCLUDE_DIR="${1#*=}"
        BUILD_ARGS="${BUILD_ARGS} --config-setting=cmake.define.Boost_INCLUDE_DIR='${BOOST_INCLUDE_DIR}'"
        ;;
    --boost-librarydir|--boost-librarydir=*)
        if [[ "$1" != *=* ]];
            then shift;
        fi
        BOOST_LIBRARY_DIR="${1#*=}"
        BUILD_ARGS="${BUILD_ARGS} --config-setting=cmake.define.BOOST_LIBRARYDIR='${BOOST_LIBRARY_DIR}'"
        ;;
    --boost-root|--boost-root=*)
        if [[ "$1" != *=* ]];
            then shift;
        fi
        BOOST_ROOT="${1#*=}"
        BUILD_ARGS="${BUILD_ARGS} --config-setting=cmake.define.Boost_ROOT='${BOOST_ROOT}'"
        ;;
    --opencl-include-dir|--opencl-include-dir=*)
        if [[ "$1" != *=* ]];
            then shift;
        fi
        OPENCL_INCLUDE_DIR="${1#*=}"
        BUILD_ARGS="${BUILD_ARGS} --config-setting=cmake.define.OpenCL_INCLUDE_DIR='${OPENCL_INCLUDE_DIR}'"
        ;;
    --opencl-library|--opencl-library=*)
        if [[ "$1" != *=* ]];
            then shift;
        fi
        OPENCL_LIBRARY="${1#*=}"
        BUILD_ARGS="${BUILD_ARGS} --config-setting=cmake.define.OpenCL_LIBRARY='${OPENCL_LIBRARY}'"
        ;;
    #########
    # flags #
    #########
    --bit32)
        export CMAKE_GENERATOR="Visual Studio 17 2022"
        export CMAKE_GENERATOR_PLATFORM="Win32"
        echo "[INFO] Attempting to build 32-bit version of LightGBM, which is only supported on Windows with generator '${CMAKE_GENERATOR}'."
        ;;
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
        export CMAKE_GENERATOR='MinGW Makefiles'
        # ref: https://stackoverflow.com/a/45104058/3986677
        BUILD_ARGS="${BUILD_ARGS} --config-setting=cmake.define.CMAKE_SH=CMAKE_SH-NOTFOUND"
        ;;
    --mpi)
        BUILD_ARGS="${BUILD_ARGS} --config-setting=cmake.define.USE_MPI=ON"
        ;;
    --no-isolation)
        BUILD_ARGS="${BUILD_ARGS} --no-isolation"
        PIP_INSTALL_ARGS="${PIP_INSTALL_ARGS} --no-build-isolation"
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
        exit 1
        ;;
  esac
  shift
done

pip install --prefer-binary 'build>=0.10.0'

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

    cp LICENSE ./lightgbm-python/
    cp VERSION.txt ./lightgbm-python/lightgbm/VERSION.txt

    cp -R ./cmake ./lightgbm-python
    cp CMakeLists.txt ./lightgbm-python
    cp -R ./include ./lightgbm-python
    cp -R ./src ./lightgbm-python
    cp -R ./swig ./lightgbm-python
    cp -R ./windows ./lightgbm-python
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
            ./include \
            ./src \
            ./swig \
            ./windows
        # use regular-old setuptools for these builds, to avoid
        # trying to recompile the shared library
        sed -i.bak -e '/start:build-system/,/end:build-system/d' pyproject.toml
        echo '[build-system]' >> ./pyproject.toml
        echo 'requires = ["setuptools"]' >> ./pyproject.toml
        echo 'build-backend = "setuptools.build_meta"' >> ./pyproject.toml
        echo "" >> ./pyproject.toml
        echo "recursive-include lightgbm *.dll *.so" > ./MANIFEST.in
        echo "" >> ./MANIFEST.in
        mkdir -p ./lightgbm/lib
        if test -f ../lib_lightgbm.so; then
            echo "found pre-compiled lib_lightgbm.so"
            cp ../lib_lightgbm.so ./lightgbm/lib/lib_lightgbm.so
        elif test -f ../Release/lib_lightgbm.dll; then
            echo "found pre-compiled Release/lib_lightgbm.dll"
            cp ../Release/lib_lightgbm.dll ./lightgbm/lib/lib_lightgbm.dll
        elif test -f ../windows/x64/DLL/lib_lightgbm.dll; then
            echo "found pre-compiled windows/x64/DLL/lib_lightgbm.dll"
            cp ../windows/x64/DLL/lib_lightgbm.dll ./lightgbm/lib/lib_lightgbm.dll
            cp ../windows/x64/DLL/lib_lightgbm.lib ./lightgbm/lib/lib_lightgbm.lib
        fi
        rm -f ./*.bak
    else
        BUILD_SDIST="false"
        BUILD_WHEEL="true"
    fi
fi

if test "${BUILD_SDIST}" = true; then
    echo "--- building sdist ---"
    rm -f ../dist/*.tar.gz
    python -m build \
        --sdist \
        --outdir ../dist \
        ${BUILD_ARGS} \
        .
fi

if test "${BUILD_WHEEL}" = true; then
    echo "--- building wheel ---"
    rm -f ../dist/*.whl || true
    python -m build \
        --wheel \
        --outdir ../dist \
        ${BUILD_ARGS} \
        .
fi

if test "${INSTALL}" = true; then
    echo "--- installing lightgbm ---"
    # ref for use of '--find-links': https://stackoverflow.com/a/52481267/3986677
    cd ../dist
    pip install \
        ${PIP_INSTALL_ARGS} \
        --force-reinstall \
        --no-cache-dir \
        --find-links=. \
        lightgbm
    cd ../
fi

echo "cleaning up"
rm -rf ./lightgbm-python
