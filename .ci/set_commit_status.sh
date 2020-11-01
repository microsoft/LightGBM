#!/bin/bash

# [description]
#     Set a status with a given name to the latest commit in a current PR git branch.
#
# [usage]
#     set_commit_status.sh <NAME> <STATUS>
#
# NAME: Name of status.
#       Status with existing name overwrites a previous one.
#
# STATUS: Status to be set.
#         Can be "error", "failure", "pending" or "success".

set -e

if [ -z "$GITHUB_ACTIONS" ]; then
  echo "Must be run inside GitHub Actions CI"
  exit -1
fi

if [ $# -ne 2 ]; then
  echo "Usage: $0 <NAME> <STATUS>"
  exit -1
fi

name=$1

status=$2
status=${status/cancelled/failure}
status=${status/timed_out/failure}
status=${status/in_progress/pending}
status=${status/queued/pending}

data=$(jq -n \
  --arg state $status \
  --arg url "$GITHUB_SERVER_URL/${{ github.repository }}/actions/runs/${{ github.run_id }}" \
  --arg name $name \
  '{"state":$state,"target_url":$url,"context":$name}')

curl \
  -X POST \
  -H "Accept: application/vnd.github.v3+json" \
  -H "Authorization: token ${{ secrets.WORKFLOW }}" \
  -d "$data" \
  "$GITHUB_API_URL/repos/${{ github.repository }}/statuses/${{ github.event.pull_request.head.sha }}"
