#!/bin/bash
set -e -u -o pipefail

CHECKS_OUTPUT_DIR=/tmp/lgb-revdepchecks
mkdir -p "${CHECKS_OUTPUT_DIR}"

# Pre-install all of lightgbm's reverse dependencies, and all of their dependencies,
# preferring precompiled binaries where available.
#
# This is done for speed... tools::check_packages_in_dir() only performs source
# installs of all packages, which results in lots of compilation.
#
# ref: https://github.com/wch/r-source/blob/594b842678e932088b16ec0cd3c39714a141eed9/src/library/tools/R/checktools.R#L295
#
# {lightgbm} checks do not need to care about that... as of this writing, nothing has {lightgbm} as a
# 'LinkingTo' dependency or otherwise needs {lightgbm} at build time.
Rscript ./.ci/download-r-revdeps.R "${CHECKS_OUTPUT_DIR}"

# build and install 'lightgbm'
sh ./build-cran-package.sh --no-build-vignettes
R CMD INSTALL --with-keep.source ./lightgbm_*.tar.gz

# run 'R CMD check' on lightgbm's reverse dependencies
Rscript ./.ci/run-r-revdepchecks.R "${CHECKS_OUTPUT_DIR}"

# R CMD check --no-manual --run-dontrun --run-donttest ${CHECKS_OUTPUT_DIR}/EIX_*.tar.gz || true
# R CMD check --no-manual --run-dontrun --run-donttest ${CHECKS_OUTPUT_DIR}/SHAPforxgboost_*.tar.gz || true
# R CMD check --no-manual --run-dontrun --run-donttest ${CHECKS_OUTPUT_DIR}/cbl_*.tar.gz || true
# R CMD check --no-manual --run-dontrun --run-donttest ${CHECKS_OUTPUT_DIR}/bonsai_*.tar.gz || true
# R CMD check --no-manual --run-dontrun --run-donttest ${CHECKS_OUTPUT_DIR}/fastshap_*.tar.gz || true
# R CMD check --no-manual --run-dontrun --run-donttest ${CHECKS_OUTPUT_DIR}/fastml_*.tar.gz || true
# R CMD check --no-manual --run-dontrun --run-donttest ${CHECKS_OUTPUT_DIR}/fastshap_*.tar.gz || true
# R CMD check --no-manual --run-dontrun --run-donttest ${CHECKS_OUTPUT_DIR}/predhy.GUI_*.tar.gz || true
# qeML vignettes take a very very long time to run
#R CMD check --no-manual --run-dontrun --run-donttest --ignore-vignettes ${CHECKS_OUTPUT_DIR}/qeML_*.tar.gz || true
# R CMD check --no-manual --run-dontrun --run-donttest ${CHECKS_OUTPUT_DIR}/mllrns_*.tar.gz || true
# R CMD check --no-manual --run-dontrun --run-donttest ${CHECKS_OUTPUT_DIR}/predhy_*.tar.gz || true
# R CMD check --no-manual --run-dontrun --run-donttest ${CHECKS_OUTPUT_DIR}/r2pmml_*.tar.gz || true
# R CMD check --no-manual --run-dontrun --run-donttest ${CHECKS_OUTPUT_DIR}/stackgbm_*.tar.gz || true
# R CMD check --no-manual --run-dontrun --run-donttest ${CHECKS_OUTPUT_DIR}/vip_*.tar.gz || true
