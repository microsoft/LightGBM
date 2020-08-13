#!/bin/bash

# base name of the bench
# it reads $1.out
# and generates $1.pdf
WHAT=$1
bench=$2
settings_file=$3

header="rev "
while read line
do
  if [ ! -z '$line' ]; then
    header="$header  \"$line\""
  fi
done < $settings_file

echo $header > $WHAT.out.header
cat $WHAT.out >> $WHAT.out.header


echo "set title '$WHAT'" > $WHAT.gnuplot
echo "set key autotitle columnhead outside " >> $WHAT.gnuplot
echo "set xtics rotate 1" >> $WHAT.gnuplot

echo "set term pdf color rounded enhanced fontscale 0.35 size 7in,5in" >> $WHAT.gnuplot
echo set output "'"$WHAT.pdf"'" >> $WHAT.gnuplot

col=`cat $settings_file | wc -l`
echo "plot for [col=2:$col+1] '$WHAT.out.header' using 0:col:xticlabels(1) with lines" >> $WHAT.gnuplot
echo " " >>  $WHAT.gnuplot

gnuplot -persist < $WHAT.gnuplot

# generate a png file (thumbnail)
convert -colors 256 -background white -density 300 -resize 300  -quality 0 $WHAT.pdf -background white -flatten $WHAT.png

# clean
rm $WHAT.out.header $WHAT.gnuplot


# generate html/svg graph

echo " " > $WHAT.html
cat resources/chart_header.html > $WHAT.html
echo 'var customSettings = {"TITLE":"","SUBTITLE":"","XLABEL":"","YLABEL":""};' >> $WHAT.html
#  'data' is an array of datasets (i.e. curves), each of which is an object of the form
#  {
#    key: <name of the curve>,
#    color: <optional color of the curve>,
#    values: [{
#        r: <revision number>,
#        v: <GFlops>
#    }]
#  }
echo 'var data = [' >> $WHAT.html

col=2
while read line
do
  if [ ! -z '$line' ]; then
    header="$header  \"$line\""
    echo '{"key":"'$line'","values":[' >> $WHAT.html
    i=0
    while read line2
    do
      if [ ! -z "$line2" ]; then
        val=`echo $line2 | cut -s -f $col -d ' '`
        if [ -n "$val" ]; then # skip build failures
          echo '{"r":'$i',"v":'$val'},' >> $WHAT.html
        fi
      fi
      ((i++))
    done < $WHAT.out
    echo ']},'  >> $WHAT.html
  fi
  ((col++))
done < $settings_file
echo '];'  >> $WHAT.html

echo 'var changesets = [' >> $WHAT.html
while read line2
do
  if [ ! -z '$line2' ]; then
    echo '"'`echo $line2 | cut -f 1 -d ' '`'",' >> $WHAT.html
  fi
done < $WHAT.out
echo '];'  >> $WHAT.html

echo 'var changesets_details = [' >> $WHAT.html
while read line2
do
  if [ ! -z '$line2' ]; then
    num=`echo "$line2" | cut -f 1 -d ' '`
    comment=`grep ":$num" changesets.txt | cut -f 2 -d '#'`
    echo '"'"$comment"'",' >> $WHAT.html
  fi
done < $WHAT.out
echo '];'  >> $WHAT.html

echo 'var changesets_count = [' >> $WHAT.html
i=0
while read line2
do
  if [ ! -z '$line2' ]; then
    echo $i ',' >> $WHAT.html
  fi
  ((i++))
done < $WHAT.out
echo '];'  >> $WHAT.html

cat resources/chart_footer.html >> $WHAT.html
