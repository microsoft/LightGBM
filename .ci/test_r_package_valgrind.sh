#!/bin/bash

cd R-package/tests

RDvalgrind \
  --no-readline \
  --vanilla \
  -d valgrind \
  -f testthat.R \
  2>&1 > out.log

cat out.log | grep -E "^\=" > valgrind-logs.log

bytes_definitely_lost=$(
  cat valgrind-logs.log \
      | grep -E "definitely lost\: .*" \
      | sed 's/^.*definitely lost\: \(.*\) bytes.*$/\1/' \
      | tr -d ","
)
if [[ ${bytes_definitely_lost} -gt 0 ]]; then
    echo "valgrind found ${bytes_definitely_lost} bytes definitely lost"
    exit -1
fi

bytes_indirectly_lost=$(
    cat valgrind-logs.log \
    | grep -E "indirectly lost\: .*" \
    | sed 's/^.*indirectly lost\: \(.*\) bytes.*$/\1/' \
    | tr -d ","
)
if [[ ${bytes_indirectly_lost} -gt 0 ]]; then
    echo "valgrind found ${bytes_indirectly_lost} bytes indirectly lost"
    exit -1
fi

bytes_possibly_lost=$(
    cat valgrind-logs.log \
    | grep -E "possibly lost\: .*" \
    | sed 's/^.*possibly lost\: \(.*\) bytes.*$/\1/' \
    | tr -d ","
)
if [[ ${bytes_possibly_lost} -gt 0 ]]; then
    echo "valgrind found ${bytes_possibly_lost} bytes possibly lost"
    exit -1
fi

exit 0
