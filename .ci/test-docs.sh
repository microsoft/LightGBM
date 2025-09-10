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
    lychee_args=(
        "--config=./docs/.lychee.toml"
        "--exclude-path=(^|/)docs/.*\.rst"
        "--"
        "**/*.rst"
        "**/*.md"
        "./R-package/**/*.Rd"
        "./docs/_build/html/*.html"
    )
    lychee_github_site="^https://github\.com.*"
    lychee_exclude_list=( 
        "--exclude ^https://www\.swig\.org/download\.html"
        "--exclude ^https://github\.com.*"
    )
    # run twice to overcome https://github.com/lycheeverse/lychee/issues/1791
    lychee --include "${lychee_github_site}" --github-token "${SECRETS_WORKFLOW}" --dump "${lychee_args[@]}"
    lychee "${lychee_exclude_list[@]}" --include-fragments --dump "${lychee_args[@]}"
fi
