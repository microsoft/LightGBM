#!/bin/sh
#
# [description]
#     Update files in source control based on the content of 'VERSION.txt'.
#
# [usage]
#
#     update-version.sh

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

update_file \
    ./.appveyor.yml \
    "s|^version: .*$|version: ${LGB_VERSION}.{build}|"

update_file \
    ./python-package/pyproject.toml \
    "s|^version = \"[0-9a-z.]+\"$|version = \"${LGB_VERSION}\"|"

update_file \
    ./R-package/DESCRIPTION \
    "s|^Version: .*$|Version: ${LGB_VERSION}|"

update_file \
    ./R-package/configure.ac \
    "s|^AC_INIT.*$|AC_INIT([lightgbm], [${LGB_VERSION}], [], [lightgbm], [])|"
