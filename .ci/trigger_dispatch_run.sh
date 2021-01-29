#!/bin/bash
#
# [description]
#     Trigger manual workflow run by a dispatch event.
#
# [usage]
#     trigger_dispatch_run.sh <PR_URL> <COMMENT_ID> <DISPATCH_NAME>
#
# PR_URL: URL of pull request from which dispatch is triggering.
#
# COMMENT_ID: ID of comment that is triggering a dispatch.
#
# DISPATCH_NAME: Name of a dispatch to be triggered.

set -e

if [ -z "$GITHUB_ACTIONS" ]; then
  echo "Must be run inside GitHub Actions CI"
  exit -1
fi

if [ $# -ne 3 ]; then
  echo "Usage: $0 <PR_URL> <COMMENT_ID> <DISPATCH_NAME>"
  exit -1
fi

pr_url=$1
comment_id=$2
dispatch_name=$3

pr=$(
  curl -sL \
    -H "Accept: application/vnd.github.v3+json" \
    -H "Authorization: token $SECRETS_WORKFLOW" \
    "$pr_url"
)
data=$(
  jq -n \
    --arg event_type "$dispatch_name" \
    --arg pr_number "$(echo $pr | jq '.number')" \
    --arg pr_sha "$(echo $pr | jq '.head.sha')" \
    --arg pr_branch "$(echo $pr | jq '.head.ref')" \
    --arg comment_number "$comment_id" \
    '{"event_type":$event_type,"client_payload":{"pr_number":$pr_number,"pr_sha":$pr_sha,"pr_branch":$pr_branch,"comment_number":$comment_number}}'
)
curl -sL \
  -X POST \
  -H "Accept: application/vnd.github.v3+json" \
  -H "Authorization: token $SECRETS_WORKFLOW" \
  -d "$data" \
  "${GITHUB_API_URL}/repos/microsoft/LightGBM/dispatches"
