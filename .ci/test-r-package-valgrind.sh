#!/bin/bash

set -e -E -u -o pipefail

RDscriptvalgrind ./.ci/install-r-deps.R --test || exit 1
sh build-cran-package.sh \
  --r-executable=RDvalgrind \
  --no-build-vignettes \
  || exit 1

RDvalgrind CMD INSTALL --preclean --install-tests lightgbm_*.tar.gz || exit 1

cd R-package/tests

ALL_LOGS_FILE="out.log"
VALGRIND_LOGS_FILE="valgrind-logs.log"

RDvalgrind \
  --no-readline \
  --vanilla \
  -d "valgrind --tool=memcheck --leak-check=full --track-origins=yes --gen-suppressions=all --suppressions=../../.ci/valgrind.supp" \
  -f testthat.R \
  > ${ALL_LOGS_FILE} 2>&1 || exit 1

cat ${ALL_LOGS_FILE}

echo "writing valgrind output to ${VALGRIND_LOGS_FILE}"
cat ${ALL_LOGS_FILE} | grep -E "^\=" > ${VALGRIND_LOGS_FILE}

bytes_definitely_lost=$(
  cat ${VALGRIND_LOGS_FILE} \
      | grep -E "definitely lost\: .*" \
      | sed 's/^.*definitely lost\: \(.*\) bytes.*$/\1/' \
      | tr -d ","
)
echo "valgrind found ${bytes_definitely_lost} bytes definitely lost"
if [[ ${bytes_definitely_lost} -gt 0 ]]; then
    exit 1
fi

bytes_indirectly_lost=$(
    cat ${VALGRIND_LOGS_FILE} \
    | grep -E "indirectly lost\: .*" \
    | sed 's/^.*indirectly lost\: \(.*\) bytes.*$/\1/' \
    | tr -d ","
)
echo "valgrind found ${bytes_indirectly_lost} bytes indirectly lost"
if [[ ${bytes_indirectly_lost} -gt 0 ]]; then
    exit 1
fi

# Valgrind false positives are suppressed via .ci/valgrind.supp
# (specifically the OpenMP TLS thread initialization false positive)

# ensure 'grep --count' doesn't cause failures
set +e

echo "checking for invalid reads"
invalid_reads=$(
  cat ${VALGRIND_LOGS_FILE} \
    | grep --count -i "Invalid read"
)
if [[ ${invalid_reads} -gt 0 ]]; then
    echo "valgrind found invalid reads: ${invalid_reads}"
    exit 1
fi

echo "checking for invalid writes"
invalid_writes=$(
  cat ${VALGRIND_LOGS_FILE} \
    | grep --count -i "Invalid write"
)
if [[ ${invalid_writes} -gt 0 ]]; then
    echo "valgrind found invalid writes: ${invalid_writes}"
    exit 1
fi
