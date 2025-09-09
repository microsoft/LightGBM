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
    lychee_args=(
    "--config=./docs/.lychee.toml"
    "--exclude-path=(^|/)docs/.*\.rst"
    "**/*.rst"
    "**/*.md"
    "./R-package/**/*.Rd"
    "./docs/_build/html/*.html"
    )
    # run twice to overcome https://github.com/lycheeverse/lychee/issues/1791
    lychee --exclude="^https://github\.com" "${lychee_args[@]}" || exit 1
fi
