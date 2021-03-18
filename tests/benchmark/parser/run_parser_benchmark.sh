#!/bin/bash

set -e

nrow=20000
ncol=2000

build() {
  d=$1
  flag=$2
  echo "building $d $flag"
  test -d $d || (mkdir -p $d && cd $d && cmake $flag ..)
  pushd $d
  make
  popd
}

gen_data() {
  if [[ ! -f test.csv ]]; then
    echo "generating csv ..."
    python gen_csv.py test.csv $nrow $ncol
  fi
}

parser_benchmark() {
  echo "========== Benchmark run Atof parser =========="
  for i in {1..3}; do
#    /usr/bin/time ./build/parser test.csv $ncol
    time ./build/parser test.csv $ncol
  done

  echo
  echo "========== Benchmark run AtofPrecise parser =========="
  for i in {1..3}; do
#    /usr/bin/time ./build-precise/parser test.csv $ncol
    time ./build-precise/parser test.csv $ncol
  done
}

build build ""
build build-precise "-DUSE_PRECISE_TEXT_PARSER=on"
gen_data
parser_benchmark
