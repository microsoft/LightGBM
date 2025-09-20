#!/bin/bash

#
# ./.ci/get-workflow-status.sh 'fix/comment-triggered-jobs' 'r_valgrind.yml'
#

set -e -u -o pipefail

BRANCH="${1}"
WORKFLOW_FILE="${2}"

echo "Searching for latest run of '${WORKFLOW_FILE}' on branch '${BRANCH}'"

LATEST_RUN_ID=$(
    gh run list  \
        --repo "microsoft/LightGBM"   \
        --branch "${BRANCH}" \
        --workflow "${WORKFLOW_FILE}" \
        --json 'createdAt,databaseId' \
        --jq 'sort_by(.createdAt) | reverse | .[0] | .databaseId'
)

if [[ "${LATEST_RUN_ID}" == "" ]]; then
    echo "No runs of '${WORKFLOW_FILE}' found on branch '${BRANCH}'"
    exit 0
fi

echo "Checking status of workflow run '${LATEST_RUN_ID}'"
gh run view \
    --repo "microsoft/LightGBM" \
    --exit-status \
    "${LATEST_RUN_ID}"
