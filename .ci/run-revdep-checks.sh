#!/bin/bash
set -e -u -o pipefail

CHECKS_OUTPUT_DIR=/tmp/lgb-revdepchecks
mkdir -p "${CHECKS_OUTPUT_DIR}"

# pre-install all of lightgbm's reverse dependencies,
# and all of their dependencies
Rscript ./.ci/download-r-revdeps.R "${CHECKS_OUTPUT_DIR}"

# build and install 'lightgbm'
sh build-cran-package.sh --no-build-vignettes
R CMD INSTALL --with-keep.source ./lightgbm_*.tar.gz

# run 'R CMD check' on lightgbm's reverse dependencies
Rscript ./.ci/run-r-revdepchecks.R "${CHECKS_OUTPUT_DIR}"
