#!/bin/bash

set -e -E -u -o pipefail

pwsh -command "Install-Module -Name PSScriptAnalyzer -Scope CurrentUser -SkipPublisherCheck"
echo "Linting PowerShell code"
pwsh -file ./.ci/lint-powershell.ps1 || exit 1

conda create -q -y -n test-env \
    "python=3.13[build=*_cp*]" \
    'biome>=1.9.3' \
    'pre-commit>=3.8.0' \
    'r-lintr>=3.1.2'

# shellcheck disable=SC1091
source activate test-env

echo "Running pre-commit checks"
pre-commit run --all-files || exit 1

echo "Linting R code"
Rscript ./.ci/lint-r-code.R "$(pwd)" || exit 1

echo "Linting JavaScript code"
bash ./.ci/lint-js.sh || exit 1
