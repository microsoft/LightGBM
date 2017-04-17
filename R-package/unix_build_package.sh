cp ../include ./src/include -rf
cp ../src ./src/src -rf
rm ./src/Makevars
cp ./src/Makevars_fullcode ./src/Makevars -f
R CMD build --no-build-vignettes .