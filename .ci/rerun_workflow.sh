#!/bin/bash
#
# [description]
#     Rerun specified workflow for given pull request.
#
# [usage]
#     rerun_workflow.sh <WORKFLOW_ID> <PR_NUMBER> <PR_BRANCH>
#
# WORKFLOW_ID: Identifier (config name of ID) of a workflow to be rerun.
#
# PR_NUMBER: Number of pull request for which workflow should be rerun.
#
# PR_BRANCH: Name of pull request's branch.

set -e

if [ -z "$GITHUB_ACTIONS" ]; then
  echo "Must be run inside GitHub Actions CI"
  exit -1
fi

if [ $# -ne 3 ]; then
  echo "Usage: $0 <WORKFLOW_ID> <PR_NUMBER> <PR_BRANCH>"
  exit -1
fi

workflow_id=$1
pr_number=$2
pr_branch=$3

runs=$(
  curl -sL \
    -H "Accept: application/vnd.github.v3+json" \
    -H "Authorization: token $SECRETS_WORKFLOW" \
    "${GITHUB_API_URL}/repos/microsoft/LightGBM/actions/workflows/${workflow_id}/runs?event=pull_request&branch=${pr_branch}" | \
  jq '.workflow_runs'
)
runs=$(echo $runs | jq --arg pr_number "$pr_number" --arg pr_branch "$pr_branch" 'map(select(.event == "pull_request" and ((.pull_requests | length) != 0 and (.pull_requests[0].number | tostring) == $pr_number or .head_branch == $pr_branch)))')
runs=$(echo $runs | jq 'sort_by(.run_number) | reverse')

if [[ $(echo $runs | jq 'length') -gt 0 ]]; then
  curl -sL \
    -X POST \
    -H "Accept: application/vnd.github.v3+json" \
    -H "Authorization: token $SECRETS_WORKFLOW" \
    "${GITHUB_API_URL}/repos/microsoft/LightGBM/actions/runs/$(echo $runs | jq '.[0].id')/rerun"
fi
