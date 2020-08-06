#!/bin/bash

# set up R environment
CRAN_MIRROR="https://cloud.r-project.org/"
R_LIB_PATH=~/Rlib
mkdir -p $R_LIB_PATH
export R_LIBS=$R_LIB_PATH
export PATH="$R_LIB_PATH/R/bin:$PATH"

# Get details needed for installing R components
#
# NOTES:
#    * Linux builds on Azure use a container and don't need these details
if ! { [[ $AZURE == "true" ]] && [[ $OS_NAME == "linux" ]]; }; then
    R_MAJOR_VERSION=( ${R_VERSION//./ } )
    if [[ "${R_MAJOR_VERSION}" == "3" ]]; then
        export R_MAC_VERSION=3.6.3
        export R_LINUX_VERSION="3.6.3-1bionic"
        export R_APT_REPO="bionic-cran35/"
    elif [[ "${R_MAJOR_VERSION}" == "4" ]]; then
        export R_MAC_VERSION=4.0.2
        export R_LINUX_VERSION="4.0.2-1.1804.0"
        export R_APT_REPO="bionic-cran40/"
    else
        echo "Unrecognized R version: ${R_VERSION}"
        exit -1
    fi
fi

# installing precompiled R for Ubuntu
# https://cran.r-project.org/bin/linux/ubuntu/#installation
# adding steps from https://stackoverflow.com/a/56378217/3986677 to get latest version
#
# This only needs to get run on Travis because R environment for Linux
# used by Azure pipelines is set up in https://github.com/guolinke/lightgbm-ci-docker
if [[ $AZURE != "true" ]] && [[ $OS_NAME == "linux" ]]; then
    sudo apt-key adv \
        --keyserver keyserver.ubuntu.com \
        --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
    sudo add-apt-repository \
        "deb https://cloud.r-project.org/bin/linux/ubuntu ${R_APT_REPO}"
    sudo apt-get update
    sudo apt-get install \
        --no-install-recommends \
        -y --allow-downgrades \
            r-base-dev=${R_LINUX_VERSION} \
            texinfo \
            texlive-latex-recommended \
            texlive-fonts-recommended \
            texlive-fonts-extra \
            qpdf \
            || exit -1

    # https://github.com/r-lib/actions/issues/111
    if [[ $R_BUILD_TYPE == "cran" ]]; then
        sudo apt-get install \
            --no-install-recommends \
            -y \
                autoconf=$(cat R-package/AUTOCONF_UBUNTU_VERSION) \
                devscripts \
                || exit -1
    fi
fi

# Installing R precompiled for Mac OS 10.11 or higher
if [[ $OS_NAME == "macos" ]]; then
    if [[ $R_BUILD_TYPE == "cran" ]]; then
        brew install \
            automake \
            checkbashisms
    fi
    brew install qpdf
    brew cask install basictex
    export PATH="/Library/TeX/texbin:$PATH"
    sudo tlmgr --verify-repo=none update --self
    sudo tlmgr --verify-repo=none install inconsolata helvetic

    wget -q https://cran.r-project.org/bin/macosx/R-${R_MAC_VERSION}.pkg -O R.pkg
    sudo installer \
        -pkg $(pwd)/R.pkg \
        -target /

    # Fix "duplicate libomp versions" issue on Mac
    # by replacing the R libomp.dylib with a symlink to the one installed with brew
    if [[ $COMPILER == "clang" ]]; then
        ver_arr=( ${R_MAC_VERSION//./ } )
        R_MAJOR_MINOR="${ver_arr[0]}.${ver_arr[1]}"
        sudo ln -sf \
            "$(brew --cellar libomp)"/*/lib/libomp.dylib \
            /Library/Frameworks/R.framework/Versions/${R_MAJOR_MINOR}/Resources/lib/libomp.dylib
    fi
fi

conda install \
    -y \
    -q \
    --no-deps \
        pandoc

# Manually install Depends and Imports libraries + 'testthat'
# to avoid a CI-time dependency on devtools (for devtools::install_deps())
packages="c('data.table', 'jsonlite', 'Matrix', 'R6', 'testthat')"
if [[ $OS_NAME == "macos" ]]; then
    packages+=", type = 'binary'"
fi
Rscript --vanilla -e "install.packages(${packages}, repos = '${CRAN_MIRROR}', lib = '${R_LIB_PATH}', dependencies = c('Depends', 'Imports', 'LinkingTo'))" || exit -1

if [[ $TASK == "r-package-check-docs" ]]; then
    Rscript build_r.R || exit -1
    Rscript --vanilla -e "install.packages('roxygen2', repos = '${CRAN_MIRROR}', lib = '${R_LIB_PATH}', dependencies = c('Depends', 'Imports', 'LinkingTo'))" || exit -1
    Rscript --vanilla -e "roxygen2::roxygenize('R-package/', load = 'installed')" || exit -1
    num_doc_files_changed=$(
        git diff --name-only | grep -E "\.Rd|NAMESPACE" | wc -l
    )
    if [[ ${num_doc_files_changed} -gt 0 ]]; then
        echo "Some R documentation files have changed. Please re-generate them and commit those changes."
        echo ""
        echo "    Rscript build_r.R"
        echo "    Rscript -e \"roxygen2::roxygenize('R-package/', load = 'installed')\""
        echo ""
        exit -1
    fi
    exit 0
fi

cd ${BUILD_DIRECTORY}

PKG_TARBALL="lightgbm_${LGB_VER}.tar.gz"
LOG_FILE_NAME="lightgbm.Rcheck/00check.log"
if [[ $R_BUILD_TYPE == "cmake" ]]; then
    Rscript build_r.R --skip-install || exit -1
elif [[ $R_BUILD_TYPE == "cran" ]]; then

    # on Linux, we recreate configure in CI to test if
    # a change in a PR has changed configure.ac
    if [[ $OS_NAME == "linux" ]]; then
        cp VERSION.txt R-package/src/VERSION.txt
        cd ${BUILD_DIRECTORY}/R-package
        autoconf \
            --output configure \
                configure.ac \
        || exit -1
        cd ${BUILD_DIRECTORY}

        num_files_changed=$(
            git diff --name-only | wc -l
        )
        if [[ ${num_files_changed} -gt 0 ]]; then
            echo "'configure' in the R package has changed. Please recreate it and commit the changes."
            echo "Changed files:"
            git diff --compact-summary
            echo "See R-package/README.md for details on how to recreate this script."
            echo ""
            exit -1
        fi
    fi

    ./build-cran-package.sh || exit -1

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
        --run-dontrun \
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
    exit -1
fi

if grep -q -R "WARNING" "$LOG_FILE_NAME"; then
    echo "WARNINGS have been found by R CMD check!"
    exit -1
fi

if [[ $OS_NAME == "linux" ]] && [[ $R_BUILD_TYPE == "cran" ]]; then
    ALLOWED_CHECK_NOTES=2
else
    ALLOWED_CHECK_NOTES=1
fi
NUM_CHECK_NOTES=$(
    cat ${LOG_FILE_NAME} \
        | grep -e '^Status: .* NOTE.*' \
        | sed 's/[^0-9]*//g'
)
if [[ ${NUM_CHECK_NOTES} -gt ${ALLOWED_CHECK_NOTES} ]]; then
    echo "Found ${NUM_CHECK_NOTES} NOTEs from R CMD check. Only ${ALLOWED_CHECK_NOTES} are allowed"
    exit -1
fi

# this check makes sure that CI builds of the CRAN package on Mac
# actually use OpenMP
if [[ $OS_NAME == "macos" ]] && [[ $R_BUILD_TYPE == "cran" ]]; then
    omp_working=$(
        cat $BUILD_LOG_FILE \
        | grep -E "checking whether OpenMP will work .*yes" \
        | wc -l
    )
    if [[ $omp_working -ne 1 ]]; then
        echo "OpenMP was not found, and should be when testing the CRAN package on macOS"
        exit -1
    fi
fi
