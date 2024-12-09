#!/bin/bash
set -e -u -o pipefail

CHECKS_OUTPUT_DIR=/tmp/lgb-revdepchecks
mkdir -p "${CHECKS_OUTPUT_DIR}"

# Pre-install all of lightgbm's reverse dependencies, and all of their dependencies,
# preferring precompiled binaries where available.
#
# This is done for speed... tools::check_packages_in_dir() only performs source
# installs of all packages, which results in lots of compilation. {lightgbm} checks
# do not need to care about that... as of this writing, nothing has {lightgbm} as a
# 'LinkingTo' dependency or otherwise needs {lightgbm} at build time.
Rscript ./.ci/download-r-revdeps.R "${CHECKS_OUTPUT_DIR}"

# build and install 'lightgbm'
sh ./build-cran-package.sh --no-build-vignettes
R CMD INSTALL --with-keep.source ./lightgbm_*.tar.gz

# run 'R CMD check' on lightgbm's reverse dependencies
Rscript ./.ci/run-r-revdepchecks.R "${CHECKS_OUTPUT_DIR}"
