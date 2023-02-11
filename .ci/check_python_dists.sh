#!/bin/sh

DIST_DIR=${1}

echo "checking Python package distributions in '${DIST_DIR}'"

pip install \
    -qq \
    check-wheel-contents \
    twine || exit -1

echo "twine check..."
twine check --strict ${DIST_DIR}/* || exit -1

echo "check-wheel-contents..."
check-wheel-contents ${DIST_DIR}/* || exit -1

echo "done checking Python package distributions"
