#!/bin/sh

# [description]
#     Prepare a source distribution of the R package
#     to be submitted to CRAN.
#
# [usage]
#     sh build-cran-package.sh

set -e

ORIG_WD=$(pwd)
TEMP_R_DIR=$(pwd)/lightgbm_r

if test -d ${TEMP_R_DIR}; then
    rm -r ${TEMP_R_DIR}
fi
mkdir -p ${TEMP_R_DIR}

 # move relevant files
cp -R R-package/* ${TEMP_R_DIR}
cp -R include ${TEMP_R_DIR}/src/
cp -R src/* ${TEMP_R_DIR}/src/

cd ${TEMP_R_DIR}

    # recreate configure script if autoconf is available,
    # otherwise skip it 
    echo "Creating 'configure' script with Autoconf"
    autoconf \
        --output configure \
        configure.ac \
        || echo "warning: not recreating configure from configure.ac"

    rm -r autom4te.cache || echo "no autoconf cache found"

    # Remove files not needed for CRAN
    echo "Removing files not needed for CRAN"
    rm src/install.libs.R
    rm -r src/cmake/
    rm -r inst/
    rm -r pkgdown/

    # main.cpp is used to make the lightgbm CLI, unnecessary
    # for the R package
    rm src/main.cpp

    # Remove 'region' and 'endregion' pragmas. This won't change
    # the correctness of the code. CRAN does not allow you
    # to use compiler flag '-Wno-unknown-pragmas' or
    # pragmas that suppress warnings.
    echo "Removing unknown pragmas in headers"
    for file in src/include/LightGBM/*.h; do
      sed \
        -i.bak \
        -e 's/^.*#pragma region.*$//' \
        -e 's/^.*#pragma endregion.*$//' \
        "${file}"
    done
    rm src/include/LightGBM/*.h.bak

    # When building an R package with 'configure', it seems
    # you're guaranteed to get a shared library called
    #  <packagename>.so/dll. The package source code expects
    # 'lib_lightgbm.so', not 'lightgbm.so', to comply with the way
    # this project has historically handled installatioon
    echo "Changing lib_lightgbm to lightgbm"
    for file in R/*.R; do
        sed \
            -i.bak \
            -e 's/lib_lightgbm/lightgbm/' \
            "${file}"
    done
    sed \
        -i.bak \
        -e 's/lib_lightgbm/lightgbm/' \
        NAMESPACE

    # 'processx' is listed as a 'Suggests' dependency in DESCRIPTION
    # because it is used in install.libs.R, a file that is not
    # included in the CRAN distribution of the pagkage
    sed \
        -i.bak \
        '/processx/d' \
        DESCRIPTION

    echo "Cleaning sed backup files"
    rm R/*.R.bak
    rm NAMESPACE.bak

cd ${ORIG_WD}

R CMD build \
    --keep-empty-dirs \
    lightgbm_r

echo "Done building R package"
