#!/bin/bash

set -e -E -u -o pipefail

conda env create \
    --name test-env \
    --file ./docs/env.yml \
|| exit 1

# shellcheck disable=SC1091
source activate test-env

# build docs
make -C docs html || exit 1

if [[ $TASK == "check-links" ]]; then
    # check docs for broken links
    pip install 'linkchecker>=10.5.0'
    linkchecker --config=./docs/.linkcheckerrc ./docs/_build/html/*.html || exit 1
fi
