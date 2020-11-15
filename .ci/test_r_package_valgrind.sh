#!/bin/bash

cd R-package/tests

ALL_LOGS_FILE="out.log"
VALGRIND_LOGS_FILE="valgrind-logs.log"

RDvalgrind \
  --no-readline \
  --vanilla \
  -d "valgrind --tool=memcheck --leak-check=full --track-origins=yes" \
  -f testthat.R \
  2>&1 > ${ALL_LOGS_FILE} || exit -1

cat ${ALL_LOGS_FILE}

cat ${ALL_LOGS_FILE} | grep -E "^\=" > ${VALGRIND_LOGS_FILE}

bytes_definitely_lost=$(
  cat ${VALGRIND_LOGS_FILE} \
      | grep -E "definitely lost\: .*" \
      | sed 's/^.*definitely lost\: \(.*\) bytes.*$/\1/' \
      | tr -d ","
)
if [[ ${bytes_definitely_lost} -gt 0 ]]; then
    echo "valgrind found ${bytes_definitely_lost} bytes definitely lost"
    exit -1
fi

bytes_indirectly_lost=$(
    cat ${VALGRIND_LOGS_FILE} \
    | grep -E "indirectly lost\: .*" \
    | sed 's/^.*indirectly lost\: \(.*\) bytes.*$/\1/' \
    | tr -d ","
)
if [[ ${bytes_indirectly_lost} -gt 0 ]]; then
    echo "valgrind found ${bytes_indirectly_lost} bytes indirectly lost"
    exit -1
fi

# one error caused by a false positive between valgrind and openmp is allowed
# ==1312== 352 bytes in 1 blocks are possibly lost in loss record 146 of 2,458
# ==1312==    at 0x483DD99: calloc (in /usr/lib/x86_64-linux-gnu/valgrind/vgpreload_memcheck-amd64-linux.so)
# ==1312==    by 0x40149CA: allocate_dtv (dl-tls.c:286)
# ==1312==    by 0x40149CA: _dl_allocate_tls (dl-tls.c:532)
# ==1312==    by 0x5702322: allocate_stack (allocatestack.c:622)
# ==1312==    by 0x5702322: pthread_create@@GLIBC_2.2.5 (pthread_create.c:660)
# ==1312==    by 0x56D0DDA: ??? (in /usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0)
# ==1312==    by 0x56C88E0: GOMP_parallel (in /usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0)
# ==1312==    by 0x154351B8: LGBM_DatasetCreateFromCSC (c_api.cpp:1286)
# ==1312==    by 0x1545789C: LGBM_DatasetCreateFromCSC_R (lightgbm_R.cpp:91)
# ==1312==    by 0x4941E2F: R_doDotCall (dotcode.c:634)
# ==1312==    by 0x494CCC6: do_dotcall (dotcode.c:1281)
# ==1312==    by 0x499FB01: bcEval (eval.c:7078)
# ==1312==    by 0x498B67F: Rf_eval (eval.c:727)
# ==1312==    by 0x498E414: R_execClosure (eval.c:1895)
bytes_possibly_lost=$(
    cat ${VALGRIND_LOGS_FILE} \
    | grep -E "possibly lost\: .*" \
    | sed 's/^.*possibly lost\: \(.*\) bytes.*$/\1/' \
    | tr -d ","
)
if [[ ${bytes_possibly_lost} -gt 352 ]]; then
    echo "valgrind found ${bytes_possibly_lost} bytes possibly lost"
    exit -1
fi

invalid_reads=$(
  cat ${VALGRIND_LOGS_FILE} \
    | grep --count -i "Invalid read"
)
if [[ ${invalid_reads} -gt 0 ]]; then
    echo "valgrind found invalid reads: ${invalid_reads}"
    exit -1
fi

invalid_writes=$(
  cat ${VALGRIND_LOGS_FILE} \
    | grep --count -i "Invalid write"
)
if [[ ${invalid_writes} -gt 0 ]]; then
    echo "valgrind found invalid writes: ${invalid_writes}"
    exit -1
fi
