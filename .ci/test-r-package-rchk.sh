#!/bin/bash

set -E -e -u -o pipefail

echo "Checking R package with rchk"
sh build-cran-package.sh --no-build-vignettes

mkdir -p ./packages
PKG_TARBALL="lightgbm_*.tar.gz"
cp ./${PKG_TARBALL} ./packages/${PKG_TARBALL}

RCHK_LOG_FILE="rchk-logs.txt"
docker run \
    --rm \
    -v $(pwd)/packages:/rchk/packages \
    -it kalibera/rchk:latest \
    "/rchk/packages/${PKG_TARBALL}" \
    2>&1 > ${RCHK_LOG_FILE} \
|| (cat ${RCHK_LOG_FILE} && exit 1)

# the exceptions below are from R itself and not LightGBM:
# https://github.com/kalibera/rchk/issues/22#issuecomment-656036156
# exit $(
#     cat ${RCHK_LOG_FILE} \
#     | grep -v "in function strptime_internal" \
#     | grep -v "in function RunGenCollect" \
#     | grep --count -E '\[PB\]|ERROR'
# )