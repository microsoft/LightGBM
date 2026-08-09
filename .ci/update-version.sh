#!/bin/bash

set -e -u -o pipefail

SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(dirname "${SCRIPT_DIR}")
LGB_VERSION=$(sed 's/rc/-/g' < "${REPO_ROOT}/VERSION.txt")

if test -z "${LGB_VERSION}"; then
    echo "VERSION.txt must contain a version" >&2
    exit 1
fi

sed_in_place() {
    if sed --version >/dev/null 2>&1; then
        sed -E -i "$@"
    else
        sed -E -i '' "$@"
    fi
}

DESCRIPTION_FILE="${REPO_ROOT}/R-package/DESCRIPTION"
CONFIGURE_AC_FILE="${REPO_ROOT}/R-package/configure.ac"

sed_in_place \
    -e "s|^(Version: ).*$|\\1${LGB_VERSION}|" \
    "${DESCRIPTION_FILE}"
sed_in_place \
    -e "s|^(AC_INIT\\(\\[lightgbm\\], \\[)[^]]*(\\].*)$|\\1${LGB_VERSION}\\2|" \
    "${CONFIGURE_AC_FILE}"

grep -Fqx "Version: ${LGB_VERSION}" "${DESCRIPTION_FILE}"
grep -Fqx \
    "AC_INIT([lightgbm], [${LGB_VERSION}], [], [lightgbm], [])" \
    "${CONFIGURE_AC_FILE}"
