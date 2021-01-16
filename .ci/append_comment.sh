#!/bin/bash
#
# [description]
#     Update comment appending a given body to the specified original comment.
#
# [usage]
#     append_comment.sh <COMMENT_ID> <BODY>
#
# COMMENT_ID: ID of comment that should be modified.
#
# BODY: Text that will be appended to the original comment body.

set -e

if [ -z "$GITHUB_ACTIONS" ]; then
  echo "Must be run inside GitHub Actions CI"
  exit -1
fi

if [ $# -ne 2 ]; then
  echo "Usage: $0 <COMMENT_ID> <BODY>"
  exit -1
fi

comment_id=$1
body=$2

old_comment_body=$(
  curl -sL \
    -H "Accept: application/vnd.github.v3+json" \
    -H "Authorization: token $SECRETS_WORKFLOW" \
    "${GITHUB_API_URL}/repos/microsoft/LightGBM/issues/comments/$comment_id" | \
  jq '.body'
)
body=${body/failure/failure ❌}
body=${body/error/failure ❌}
body=${body/cancelled/failure ❌}
body=${body/timed_out/failure ❌}
body=${body/success/success ✔️}
data=$(
  jq -n \
    --argjson body "${old_comment_body%?}\r\n\r\n$body\"" \
    '{"body":$body}'
)
curl -sL \
  -X PATCH \
  -H "Accept: application/vnd.github.v3+json" \
  -H "Authorization: token $SECRETS_WORKFLOW" \
  -d "$data" \
  "${GITHUB_API_URL}/repos/microsoft/LightGBM/issues/comments/$comment_id"
