#!/bin/bash

set -e -u -o pipefail

COMMIT="${1}"
OUTPUT_DIR="./release-artifacts"

get-latest-run-id() {
     gh run list                      \
        --repo "microsoft/LightGBM"   \
        --commit "${1}"               \
        --workflow "${2}"             \
        --json 'createdAt,databaseId' \
        --jq 'sort_by(.createdAt) | reverse | .[0] | .databaseId'
}

# ensure directory for storing artifacts exists
echo "preparing to dowload artifacts to '${OUTPUT_DIR}'"
mkdir -p "${OUTPUT_DIR}"

# get python-package artifacts
echo "downloading python-package artifacts"
gh run download \
    --repo "microsoft/LightGBM" \
    --dir "${OUTPUT_DIR}" \
    $(get-latest-run-id ${COMMIT} "python_package.yml")
echo "done downloading python-package artifacts"

echo "downloaded the following:"
find "${OUTPUT_DIR}" -type f
