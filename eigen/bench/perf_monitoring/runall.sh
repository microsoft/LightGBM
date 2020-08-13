#!/bin/bash

# ./runall.sh "Title"

# Examples of environment variables to be set:
#   PREFIX="haswell-fma-"
#   CXX_FLAGS="-mfma"
#   CXX=clang++

# Options:
#   -up : enforce the recomputation of existing data, and keep best results as a merging strategy
#   -s  : recompute selected changesets only and keep bests
#   -np : no plotting of results, just generate the data

if [[ "$*" =~ '-np' ]]; then
  do_plot=false
else
  do_plot=true
fi

./run.sh gemm gemm_settings.txt $*
./run.sh lazy_gemm lazy_gemm_settings.txt $*
./run.sh gemv gemv_settings.txt $*
./run.sh gemvt gemv_settings.txt $*
./run.sh trmv_up gemv_square_settings.txt $*
./run.sh trmv_lo gemv_square_settings.txt $*
./run.sh trmv_upt gemv_square_settings.txt $*
./run.sh trmv_lot gemv_square_settings.txt $*
./run.sh llt gemm_square_settings.txt $*

if $do_plot ; then

# generate html file

function print_td {
  echo '<td><a href="'$PREFIX'-'$1"$2"'.html"><img src="'$PREFIX'-'$1"$2"'.png" title="'$3'"></a></td>' >> $htmlfile
}

function print_tr {
  echo '<tr><th colspan="3">'"$2"'</th></tr>' >> $htmlfile
  echo '<tr>' >> $htmlfile
  print_td s $1 float
  print_td d $1 double
  print_td c $1 complex
  echo '</tr>' >> $htmlfile
}

if [ -n "$PREFIX" ]; then


cp resources/s1.js $PREFIX/
cp resources/s2.js $PREFIX/

htmlfile="$PREFIX/index.html"
cat resources/header.html > $htmlfile

echo '<h1>'$1'</h1>' >> $htmlfile
echo '<table>' >> $htmlfile
print_tr gemm       'C += A &middot; B   &nbsp; (gemm)'
print_tr lazy_gemm  'C += A &middot; B   &nbsp; (gemm lazy)'
print_tr gemv       'y += A &middot; x   &nbsp; (gemv)'
print_tr gemvt      'y += A<sup>T</sup> &middot; x  &nbsp; (gemv)'
print_tr trmv_up    'y += U &middot; x   &nbsp; (trmv)'
print_tr trmv_upt   'y += U<sup>T</sup> &middot; x  &nbsp; (trmv)'
print_tr trmv_lo    'y += L &middot; x   &nbsp; (trmv)'
print_tr trmv_lot   'y += L<sup>T</sup> &middot; x  &nbsp; (trmv)'
print_tr trmv_lot   'L &middot; L<sup>T<sup> = A &nbsp;  (Cholesky,potrf)'

cat resources/footer.html >> $htmlfile

fi
fi
