#!/bin/bash
# https://github.com/r-lib/actions/issues/332#issuecomment-878271645
set -e -u -o pipefail

CHECKS_OUTPUT_DIR=/tmp/lgb-revdepchecks2
mkdir -p "${CHECKS_OUTPUT_DIR}"

sh build-cran-package.sh --no-build-vignettes
mv ./lightgbm_*.tar.gz "${CHECKS_OUTPUT_DIR}/"

Rscript ./.ci/run-r-revdepchecks.R "${CHECKS_OUTPUT_DIR}"
