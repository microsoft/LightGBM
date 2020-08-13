#!/bin/bash

# ./run.sh gemm gemm_settings.txt
# ./run.sh lazy_gemm lazy_gemm_settings.txt
# ./run.sh gemv gemv_settings.txt
# ./run.sh trmv_up gemv_square_settings.txt
# ...

# Examples of environment variables to be set:
#   PREFIX="haswell-fma-"
#   CXX_FLAGS="-mfma"
#   CXX=clang++

# Options:
#   -up : enforce the recomputation of existing data, and keep best results as a merging strategy
#   -s  : recompute selected changesets only and keep bests
#   -np : no plotting of results, just generate the data

bench=$1
settings_file=$2

if [[ "$*" =~ '-up' ]]; then
  update=true
else
  update=false
fi

if [[ "$*" =~ '-s' ]]; then
  selected=true
else
  selected=false
fi

if [[ "$*" =~ '-np' ]]; then
  do_plot=false
else
  do_plot=true
fi


WORKING_DIR=${PREFIX:?"default"}

if [ -z "$PREFIX" ]; then
  WORKING_DIR_PREFIX="$WORKING_DIR/"
else
  WORKING_DIR_PREFIX="$WORKING_DIR/$PREFIX-"
fi
echo "WORKING_DIR_PREFIX=$WORKING_DIR_PREFIX"
mkdir -p $WORKING_DIR

global_args="$*"

if $selected ; then
 echo "Recompute selected changesets only and keep bests"
elif $update ; then
 echo "(Re-)Compute all changesets and keep bests"
else
 echo "Skip previously computed changesets"
fi



if [ ! -d "eigen_src" ]; then
  git clone https://gitlab.com/libeigen/eigen.git eigen_src
else
  cd eigen_src
  git pull
  cd ..
fi

if [ -z "$CXX" ]; then
  CXX=g++
fi

function make_backup
{
  if [ -f "$1.out" ]; then
    mv "$1.out" "$1.backup"
  fi
}

function merge
{
  count1=`echo $1 |  wc -w`
  count2=`echo $2 |  wc -w`
  
  if [ $count1 == $count2 ]; then
    a=( $1 ); b=( $2 )
    res=""
    for (( i=0 ; i<$count1 ; i++ )); do
      ai=${a[$i]}; bi=${b[$i]}
      tmp=`echo "if ($ai > $bi) $ai else $bi " | bc -l`
      res="$res $tmp"
    done
    echo $res

  else
    echo $1
  fi
}

function test_current 
{
  rev=$1
  scalar=$2
  name=$3
  
  prev=""
  if [ -e "$name.backup" ]; then
    prev=`grep $rev "$name.backup" | cut -d ' ' -f 2-`
  fi
  res=$prev
  count_rev=`echo $prev |  wc -w`
  count_ref=`cat $settings_file |  wc -l`
  if echo "$global_args" | grep "$rev" > /dev/null; then
    rev_found=true
  else
    rev_found=false
  fi
#  echo $update et $selected et $rev_found because $rev et "$global_args"
#  echo $count_rev et $count_ref
  if $update || [ $count_rev != $count_ref ] || ( $selected &&  $rev_found ); then
    echo "RUN: $CXX -O3 -DNDEBUG -march=native $CXX_FLAGS -I eigen_src $bench.cpp -DSCALAR=$scalar -o $name"
    if $CXX -O3 -DNDEBUG -march=native $CXX_FLAGS -I eigen_src $bench.cpp -DSCALAR=$scalar -o $name; then
      curr=`./$name $settings_file`
      if [ $count_rev == $count_ref ]; then
        echo "merge previous $prev"
        echo "with new       $curr"
      else
        echo "got            $curr"
      fi
      res=`merge "$curr" "$prev"`
#       echo $res
      echo "$rev $res" >> $name.out
    else
      echo "Compilation failed, skip rev $rev"
    fi
  else
    echo "Skip existing results for $rev / $name"
    echo "$rev $res" >> $name.out
  fi
}

make_backup $WORKING_DIR_PREFIX"s"$bench
make_backup $WORKING_DIR_PREFIX"d"$bench
make_backup $WORKING_DIR_PREFIX"c"$bench

cut -f1 -d"#" < changesets.txt | grep -E '[[:alnum:]]' | while read rev
do
  if [ ! -z '$rev' ]; then
    rev2=`echo $rev | cut -f 2 -d':'`
    echo "Testing rev $rev, $rev2"
    cd eigen_src
    git checkout $rev2 > /dev/null
    actual_rev=`git rev-parse --short HEAD`
    cd ..
    
    test_current $actual_rev float                  $WORKING_DIR_PREFIX"s"$bench
    test_current $actual_rev double                 $WORKING_DIR_PREFIX"d"$bench
    test_current $actual_rev "std::complex<double>" $WORKING_DIR_PREFIX"c"$bench
  fi
  
done

echo "Float:"
cat $WORKING_DIR_PREFIX"s""$bench.out"
echo " "

echo "Double:"
cat $WORKING_DIR_PREFIX"d""$bench.out"
echo ""

echo "Complex:"
cat $WORKING_DIR_PREFIX"c""$bench.out"
echo ""

if $do_plot ; then

./make_plot.sh $WORKING_DIR_PREFIX"s"$bench $bench $settings_file
./make_plot.sh $WORKING_DIR_PREFIX"d"$bench $bench $settings_file
./make_plot.sh $WORKING_DIR_PREFIX"c"$bench $bench $settings_file

fi
