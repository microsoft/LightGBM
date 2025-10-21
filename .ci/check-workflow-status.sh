#!/bin/bash

# [description]
#
#   Look for the last run of a given GitHub Actions workflow on a given branch.
#   If there's never been one (as might be the case with optional workflows like valgrind),
#   exit with 0.
#
#   Otherwise, check the status of that latest run.
#   If it wasn't successful, exit with a non-0 exit code.
#
# [usage]
#
#     check-workflow-status.sh <BRANCH> <WORKFLOW_FILE>
#
# BRANCH: name of a branch involved in a pull request.
#
# WORKFLOW_FILE: filename (e.g. 'r_valgrind.yml') defining the GitHub Actions workflow.
#

set -e -u -o pipefail

BRANCH="${1}"
WORKFLOW_FILE="${2}"

echo "Searching for latest run of '${WORKFLOW_FILE}' on branch '${BRANCH}'"

LATEST_RUN_ID=$(
    gh run list  \
        --repo "microsoft/LightGBM" \
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
