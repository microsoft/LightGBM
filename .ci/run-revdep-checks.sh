#!/bin/bash
# https://github.com/r-lib/actions/issues/332#issuecomment-878271645
set -e -u -o pipefail

CHECKS_OUTPUT_DIR=/tmp/lgb-revdepchecks
mkdir -p "${CHECKS_OUTPUT_DIR}"

sh build-cran-package.sh --no-build-vignettes
mv ./lightgbm_*.tar.gz "${CHECKS_OUTPUT_DIR}/"

# pre-install all of the dependencies... tools::check_packages_in_dir()
# is hard-coded to compile them all from source
# (https://github.com/wch/r-source/blob/594b842678e932088b16ec0cd3c39714a141eed9/src/library/tools/R/checktools.R#L295)
Rscript ./.ci/download-r-revdeps.R

Rscript ./.ci/run-r-revdepchecks.R "${CHECKS_OUTPUT_DIR}"
