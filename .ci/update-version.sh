#!/bin/sh

set -e -u

LGB_VERSION=$(sed 's/rc/-/g' < ./VERSION.txt)

if test -z "${LGB_VERSION}"; then
    echo "VERSION.txt must contain a version" >&2
    exit 1
fi

update_file() {
    TARGET_FILE=$1
    SED_EXPRESSION=$2
    UPDATED_CONTENTS=$(sed "${SED_EXPRESSION}" "${TARGET_FILE}")
    printf '%s\n' "${UPDATED_CONTENTS}" > "${TARGET_FILE}"
}

DESCRIPTION_FILE="./R-package/DESCRIPTION"
CONFIGURE_AC_FILE="./R-package/configure.ac"

update_file \
    "${DESCRIPTION_FILE}" \
    "s|^Version: .*$|Version: ${LGB_VERSION}|"
update_file \
    "${CONFIGURE_AC_FILE}" \
    "s|^AC_INIT.*$|AC_INIT([lightgbm], [${LGB_VERSION}], [], [lightgbm], [])|"

grep -Fqx "Version: ${LGB_VERSION}" "${DESCRIPTION_FILE}"
grep -Fqx \
    "AC_INIT([lightgbm], [${LGB_VERSION}], [], [lightgbm], [])" \
    "${CONFIGURE_AC_FILE}"
