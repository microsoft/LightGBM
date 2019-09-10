#!/bin/bash

# [description]
#     Lint R code in a directory for style
#     problems
# [usage]
#     ./.ci/lint_r.sh $(pwd)

SOURCE_DIR=${1}

# failure is a natural part of life
set -e

echo ""
echo "Checking code for R style problems..."
echo ""

    Rscript $(pwd)/.ci/lint_r_code.R \
        --source-dir ${SOURCE_DIR}

echo ""
echo "Done checking code for style problems."
echo ""
