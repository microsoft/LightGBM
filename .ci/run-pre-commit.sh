#!/bin/bash

set -e -E -u -o pipefail

pre-commit run --all-files || exit 1
