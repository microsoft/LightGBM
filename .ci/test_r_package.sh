#!/bin/bash

set -e -E -u -o pipefail

# defaults
ARCH=$(uname -m)
INSTALL_CMAKE_FROM_RELEASES=${INSTALL_CMAKE_FROM_RELEASES:-"false"}

# set up R environment
CRAN_MIRROR="https://cran.rstudio.com"
R_LIB_PATH=~/Rlib
mkdir -p $R_LIB_PATH
export R_LIBS=$R_LIB_PATH
export PATH="$R_LIB_PATH/R/bin:$PATH"

# don't fail builds for long-running examples unless they're very long.
# See https://github.com/microsoft/LightGBM/issues/4049#issuecomment-793412254.
if [[ $R_BUILD_TYPE != "cran" ]]; then
    export _R_CHECK_EXAMPLE_TIMING_THRESHOLD_=30
fi

# Get details needed for installing R components
R_MAJOR_VERSION=( ${R_VERSION//./ } )
if [[ "${R_MAJOR_VERSION}" == "3" ]]; then
    export R_MAC_VERSION=3.6.3
    export R_MAC_PKG_URL=${CRAN_MIRROR}/bin/macosx/R-${R_MAC_VERSION}.nn.pkg
    export R_LINUX_VERSION="3.6.3-1bionic"
    export R_APT_REPO="bionic-cran35/"
elif [[ "${R_MAJOR_VERSION}" == "4" ]]; then
    export R_MAC_VERSION=4.3.1
    export R_MAC_PKG_URL=${CRAN_MIRROR}/bin/macosx/big-sur-${ARCH}/base/R-${R_MAC_VERSION}-${ARCH}.pkg
    export R_LINUX_VERSION="4.3.1-1.2204.0"
    export R_APT_REPO="jammy-cran40/"
else
    echo "Unrecognized R version: ${R_VERSION}"
    exit 1
fi

# installing precompiled R for Ubuntu
# https://cran.r-project.org/bin/linux/ubuntu/#installation
# adding steps from https://stackoverflow.com/a/56378217/3986677 to get latest version
#
# `devscripts` is required for 'checkbashisms' (https://github.com/r-lib/actions/issues/111)
if [[ $OS_NAME == "linux" ]]; then
    mkdir -p ~/.gnupg
    echo "disable-ipv6" >> ~/.gnupg/dirmngr.conf
    sudo apt-key adv \
        --homedir ~/.gnupg \
        --keyserver keyserver.ubuntu.com \
        --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9 || exit 1
    sudo add-apt-repository \
        "deb ${CRAN_MIRROR}/bin/linux/ubuntu ${R_APT_REPO}" || exit 1
    sudo apt-get update
    sudo apt-get install \
        --no-install-recommends \
        -y \
            devscripts \
            r-base-core=${R_LINUX_VERSION} \
            r-base-dev=${R_LINUX_VERSION} \
            texinfo \
            texlive-latex-extra \
            texlive-latex-recommended \
            texlive-fonts-recommended \
            texlive-fonts-extra \
            tidy \
            qpdf \
            || exit 1

    if [[ $R_BUILD_TYPE == "cran" ]]; then
        sudo apt-get install \
            --no-install-recommends \
            -y \
                autoconf=$(cat R-package/AUTOCONF_UBUNTU_VERSION) \
                automake \
                || exit 1
    fi
    if [[ $INSTALL_CMAKE_FROM_RELEASES == "true" ]]; then
        curl -O -L \
            https://github.com/Kitware/CMake/releases/download/v3.25.1/cmake-3.25.1-linux-${ARCH}.sh \
        || exit 1

        sudo mkdir /opt/cmake || exit 1
        sudo sh cmake-3.25.1-linux-${ARCH}.sh --skip-license --prefix=/opt/cmake || exit 1
        sudo ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake || exit 1
    fi
fi

# Installing R precompiled for Mac OS 10.11 or higher
if [[ $OS_NAME == "macos" ]]; then
    brew update-reset --auto-update
    brew update --auto-update
    if [[ $R_BUILD_TYPE == "cran" ]]; then
        brew install automake || exit 1
    fi
    brew install \
        checkbashisms \
        qpdf || exit 1
    brew install basictex || exit 1
    export PATH="/Library/TeX/texbin:$PATH"
    sudo tlmgr --verify-repo=none update --self || exit 1
    sudo tlmgr --verify-repo=none install inconsolata helvetic rsfs || exit 1

    curl -sL ${R_MAC_PKG_URL} -o R.pkg || exit 1
    sudo installer \
        -pkg $(pwd)/R.pkg \
        -target / || exit 1
fi

# fix for issue where CRAN was not returning {lattice} when using R 3.6
# "Warning: dependency ‘lattice’ is not available"
if [[ "${R_MAJOR_VERSION}" == "3" ]]; then
    Rscript --vanilla -e "install.packages('https://cran.r-project.org/src/contrib/Archive/lattice/lattice_0.20-41.tar.gz', repos = NULL, lib = '${R_LIB_PATH}')"
else
    # {Matrix} needs {lattice}, so this needs to run before manually installing {Matrix}.
    # This should be unnecessary on R >=4.4.0
    # ref: https://github.com/microsoft/LightGBM/issues/6433
    Rscript --vanilla -e "install.packages('lattice', repos = '${CRAN_MIRROR}', lib = '${R_LIB_PATH}')"
fi

# manually install {Matrix}, as {Matrix}=1.7-0 raised its R floor all the way to R 4.4.0
# ref: https://github.com/microsoft/LightGBM/issues/6433
Rscript --vanilla -e "install.packages('https://cran.r-project.org/src/contrib/Archive/Matrix/Matrix_1.6-5.tar.gz', repos = NULL, lib = '${R_LIB_PATH}')"

# Manually install Depends and Imports libraries + 'knitr', 'markdown', 'RhpcBLASctl', 'testthat'
# to avoid a CI-time dependency on devtools (for devtools::install_deps())
# NOTE: testthat is not required when running rchk
if [[ "${TASK}" == "r-rchk" ]]; then
    packages="c('data.table', 'jsonlite', 'knitr', 'markdown', 'R6', 'RhpcBLASctl')"
else
    packages="c('data.table', 'jsonlite', 'knitr', 'markdown', 'R6', 'RhpcBLASctl', 'testthat')"
fi
compile_from_source="both"
if [[ $OS_NAME == "macos" ]]; then
    packages+=", type = 'binary'"
    compile_from_source="never"
fi
Rscript --vanilla -e "options(install.packages.compile.from.source = '${compile_from_source}'); install.packages(${packages}, repos = '${CRAN_MIRROR}', lib = '${R_LIB_PATH}', dependencies = c('Depends', 'Imports', 'LinkingTo'), Ncpus = parallel::detectCores())" || exit 1

cd ${BUILD_DIRECTORY}

PKG_TARBALL="lightgbm_*.tar.gz"
LOG_FILE_NAME="lightgbm.Rcheck/00check.log"
if [[ $R_BUILD_TYPE == "cmake" ]]; then
    Rscript build_r.R -j4 --skip-install || exit 1
elif [[ $R_BUILD_TYPE == "cran" ]]; then

    # on Linux, we recreate configure in CI to test if
    # a change in a PR has changed configure.ac
    if [[ $OS_NAME == "linux" ]]; then
        ${BUILD_DIRECTORY}/R-package/recreate-configure.sh

        num_files_changed=$(
            git diff --name-only | wc -l
        )
        if [[ ${num_files_changed} -gt 0 ]]; then
            echo "'configure' in the R package has changed. Please recreate it and commit the changes."
            echo "Changed files:"
            git diff --compact-summary
            echo "See R-package/README.md for details on how to recreate this script."
            echo ""
            exit 1
        fi
    fi

    ./build-cran-package.sh || exit 1

    if [[ "${TASK}" == "r-rchk" ]]; then
        echo "Checking R package with rchk"
        mkdir -p packages
        cp ${PKG_TARBALL} packages
        RCHK_LOG_FILE="rchk-logs.txt"
        docker run \
            -v $(pwd)/packages:/rchk/packages \
            kalibera/rchk:latest \
            "/rchk/packages/${PKG_TARBALL}" \
        2>&1 > ${RCHK_LOG_FILE} \
        || (cat ${RCHK_LOG_FILE} && exit 1)
        cat ${RCHK_LOG_FILE}

        # the exceptions below are from R itself and not LightGBM:
        # https://github.com/kalibera/rchk/issues/22#issuecomment-656036156
        exit $(
            cat ${RCHK_LOG_FILE} \
            | grep -v "in function strptime_internal" \
            | grep -v "in function RunGenCollect" \
            | grep --count -E '\[PB\]|ERROR'
        )
    fi

    # Test CRAN source .tar.gz in a directory that is not this repo or below it.
    # When people install.packages('lightgbm'), they won't have the LightGBM
    # git repo around. This is to protect against the use of relative paths
    # like ../../CMakeLists.txt that would only work if you are in the repo
    R_CMD_CHECK_DIR="${HOME}/tmp-r-cmd-check/"
    mkdir -p ${R_CMD_CHECK_DIR}
    mv ${PKG_TARBALL} ${R_CMD_CHECK_DIR}
    cd ${R_CMD_CHECK_DIR}
fi

# fails tests if either ERRORs or WARNINGs are thrown by
# R CMD CHECK
check_succeeded="yes"
(
    R CMD check ${PKG_TARBALL} \
        --as-cran \
        --run-donttest \
    || check_succeeded="no"
) &

# R CMD check suppresses output, some CIs kill builds after
# a few minutes with no output. This trick gives R CMD check more time
#     * https://github.com/travis-ci/travis-ci/issues/4190#issuecomment-169987525
#     * https://stackoverflow.com/a/29890106/3986677
CHECK_PID=$!
while kill -0 ${CHECK_PID} >/dev/null 2>&1; do
    echo -n -e " \b"
    sleep 5
done

echo "R CMD check build logs:"
BUILD_LOG_FILE=lightgbm.Rcheck/00install.out
cat ${BUILD_LOG_FILE}

if [[ $check_succeeded == "no" ]]; then
    exit 1
fi

# ensure 'grep --count' doesn't cause failures
set +e

used_correct_r_version=$(
    cat $LOG_FILE_NAME \
    | grep --count "using R version ${R_VERSION}"
)
if [[ $used_correct_r_version -ne 1 ]]; then
    echo "Unexpected R version was used. Expected '${R_VERSION}'."
    exit 1
fi

if [[ $R_BUILD_TYPE == "cmake" ]]; then
    passed_correct_r_version_to_cmake=$(
        cat $BUILD_LOG_FILE \
        | grep --count "R version passed into FindLibR.cmake: ${R_VERSION}"
    )
    if [[ $used_correct_r_version -ne 1 ]]; then
        echo "Unexpected R version was passed into cmake. Expected '${R_VERSION}'."
        exit 1
    fi
fi


if grep -q -E "NOTE|WARNING|ERROR" "$LOG_FILE_NAME"; then
    echo "NOTEs, WARNINGs, or ERRORs have been found by R CMD check"
    exit 1
fi

# this check makes sure that CI builds of the package actually use OpenMP
if [[ $OS_NAME == "macos" ]] && [[ $R_BUILD_TYPE == "cran" ]]; then
    omp_working=$(
        cat $BUILD_LOG_FILE \
        | grep --count -E "checking whether OpenMP will work .*yes"
    )
elif [[ $R_BUILD_TYPE == "cmake" ]]; then
    omp_working=$(
        cat $BUILD_LOG_FILE \
        | grep --count -E ".*Found OpenMP: TRUE.*"
    )
else
    omp_working=1
fi
if [[ $omp_working -ne 1 ]]; then
    echo "OpenMP was not found"
    exit 1
fi

# this check makes sure that CI builds of the package
# actually use MM_PREFETCH preprocessor definition
if [[ $R_BUILD_TYPE == "cran" ]]; then
    mm_prefetch_working=$(
        cat $BUILD_LOG_FILE \
        | grep --count -E "checking whether MM_PREFETCH work.*yes"
    )
else
    mm_prefetch_working=$(
        cat $BUILD_LOG_FILE \
        | grep --count -E ".*Performing Test MM_PREFETCH - Success"
    )
fi
if [[ $mm_prefetch_working -ne 1 ]]; then
    echo "MM_PREFETCH test was not passed"
    exit 1
fi

# this check makes sure that CI builds of the package
# actually use MM_MALLOC preprocessor definition
if [[ $R_BUILD_TYPE == "cran" ]]; then
    mm_malloc_working=$(
        cat $BUILD_LOG_FILE \
        | grep --count -E "checking whether MM_MALLOC work.*yes"
    )
else
    mm_malloc_working=$(
        cat $BUILD_LOG_FILE \
        | grep --count -E ".*Performing Test MM_MALLOC - Success"
    )
fi
if [[ $mm_malloc_working -ne 1 ]]; then
    echo "MM_MALLOC test was not passed"
    exit 1
fi

# this check makes sure that no "warning: unknown pragma ignored" logs
# reach the user leading them to believe that something went wrong
if [[ $R_BUILD_TYPE == "cran" ]]; then
    pragma_warning_present=$(
        cat $BUILD_LOG_FILE \
        | grep --count -E "warning: unknown pragma ignored"
    )
    if [[ $pragma_warning_present -ne 0 ]]; then
        echo "Unknown pragma warning is present, pragmas should have been removed before build"
        exit 1
    fi
fi
