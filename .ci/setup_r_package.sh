#!/bin/bash

# set up R environment
export CRAN_MIRROR="https://cloud.r-project.org/"
mkdir -p $R_LIB_PATH
export R_LIBS=$R_LIB_PATH
export PATH="$R_LIB_PATH/R/bin:$PATH"

# hack to get around this:
# https://stat.ethz.ch/pipermail/r-package-devel/2020q3/005930.html
export _R_CHECK_SYSTEM_CLOCK_=0

# Get details needed for installing R components
R_MAJOR_VERSION=( ${R_VERSION//./ } )
if [[ "${R_MAJOR_VERSION}" == "3" ]]; then
    export R_MAC_VERSION=3.6.3
    export R_LINUX_VERSION="3.6.3-1bionic"
    export R_APT_REPO="bionic-cran35/"
elif [[ "${R_MAJOR_VERSION}" == "4" ]]; then
    export R_MAC_VERSION=4.0.3
    export R_LINUX_VERSION="4.0.3-1.1804.0"
    export R_APT_REPO="bionic-cran40/"
else
    echo "Unrecognized R version: ${R_VERSION}"
    exit -1
fi

# installing precompiled R for Ubuntu
# https://cran.r-project.org/bin/linux/ubuntu/#installation
# adding steps from https://stackoverflow.com/a/56378217/3986677 to get latest version
#
# `devscripts` is required for 'checkbashisms' (https://github.com/r-lib/actions/issues/111)
if [[ $OS_NAME == "linux" ]]; then
    sudo apt-key adv \
        --keyserver keyserver.ubuntu.com \
        --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
    sudo add-apt-repository \
        "deb https://cloud.r-project.org/bin/linux/ubuntu ${R_APT_REPO}"
    sudo apt-get update
    sudo apt-get install \
        --no-install-recommends \
        -y --allow-downgrades \
            devscripts \
            r-base-dev=${R_LINUX_VERSION} \
            texinfo \
            texlive-latex-recommended \
            texlive-fonts-recommended \
            texlive-fonts-extra \
            qpdf \
            || exit -1

    
    if [[ $R_BUILD_TYPE == "cran" ]]; then
        sudo apt-get install \
            --no-install-recommends \
            -y \
                autoconf=$(cat R-package/AUTOCONF_UBUNTU_VERSION) \
                || exit -1
    fi
fi

# Installing R precompiled for Mac OS 10.11 or higher
if [[ $OS_NAME == "macos" ]]; then
    if [[ $R_BUILD_TYPE == "cran" ]]; then
        brew install automake
    fi
    brew install \
        checkbashisms \
        qpdf
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
    Rscript --vanilla -e "install.packages('roxygen2', repos = '${CRAN_MIRROR}', lib = '${R_LIB_PATH}', dependencies = c('Depends', 'Imports', 'LinkingTo'))" || exit -1
fi
