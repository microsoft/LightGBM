#!/bin/bash

# This is a script example to automatically update and upload performance unit tests.
# The following five variables must be adjusted to match your settings.

USER='ggael'
UPLOAD_DIR=perf_monitoring/ggaelmacbook26
EIGEN_SOURCE_PATH=$HOME/Eigen/eigen
export PREFIX="haswell-fma"
export CXX_FLAGS="-mfma -w"

####

BENCH_PATH=$EIGEN_SOURCE_PATH/bench/perf_monitoring/$PREFIX
PREVPATH=$(pwd)
cd $EIGEN_SOURCE_PATH/bench/perf_monitoring && ./runall.sh "Haswell 2.6GHz, FMA, Apple's clang" "$@"
cd $PREVPATH || exit 1

ALLFILES="$BENCH_PATH/*.png $BENCH_PATH/*.html $BENCH_PATH/index.html $BENCH_PATH/s1.js $BENCH_PATH/s2.js"

# (the '/' at the end of path is very important, see rsync documentation)
rsync -az --no-p --delete $ALLFILES $USER@ssh.tuxfamily.org:eigen/eigen.tuxfamily.org-web/htdocs/$UPLOAD_DIR/ || { echo "upload failed"; exit 1; }

# fix the perm
ssh $USER@ssh.tuxfamily.org "chmod -R g+w /home/eigen/eigen.tuxfamily.org-web/htdocs/perf_monitoring" || { echo "perm failed"; exit 1; }
