#!/bin/bash
#
# [description]
#     Set a status with a given name to the specified commit.
#
# [usage]
#     set_commit_status.sh <NAME> <STATUS> <SHA>
#
# NAME: Name of status.
#       Status with existing name overwrites a previous one.
#
# STATUS: Status to be set.
#         Can be "error", "failure", "pending" or "success".
#
# SHA: SHA of a commit to set a status on.

set -e

if [ -z "$GITHUB_ACTIONS" ]; then
  echo "Must be run inside GitHub Actions CI"
  exit -1
fi

if [ $# -ne 3 ]; then
  echo "Usage: $0 <NAME> <STATUS> <SHA>"
  exit -1
fi

name=$1

status=$2
status=${status/error/failure}
status=${status/cancelled/failure}
status=${status/timed_out/failure}
status=${status/in_progress/pending}
status=${status/queued/pending}

sha=$3

data=$(
  jq -n \
    --arg state $status \
    --arg url "${GITHUB_SERVER_URL}/microsoft/LightGBM/actions/runs/${GITHUB_RUN_ID}" \
    --arg name "$name" \
    '{"state":$state,"target_url":$url,"context":$name}'
)

curl -sL \
  -X POST \
  -H "Accept: application/vnd.github.v3+json" \
  -H "Authorization: token $SECRETS_WORKFLOW" \
  -d "$data" \
  "${GITHUB_API_URL}/repos/microsoft/LightGBM/statuses/$sha"
