#!/bin/bash

set -e -E -u -o pipefail

biome ci --config-path=./biome.json --diagnostic-level=info --error-on-warnings ./
