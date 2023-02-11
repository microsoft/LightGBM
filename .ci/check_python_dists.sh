#!/bin/sh

DIST_DIR=${1}

echo "checking Python package distributions in '${DIST_DIR}'"

pip install \
    check-wheel-contents \
    twine || exit -1

twine check --strict "${DIST_DIR}/*" || exit -1
check-wheel-contents "${DIST_DIR}/*.whl" || exit -1
