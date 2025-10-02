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
    conda install -y -n test-env 'lychee>=0.20.1'
    # to see all gained files add "--dump-inputs" flag
    # to see all gained links add "--dump" flag
    lychee \
        "--config=./docs/.lychee.toml" \
        "--" \
        "**/*.rst" \
        "**/*.md" \
        "./R-package/**/*.Rd" \
        "./docs/_build/html/*.html" \
    || exit 1
fi
