
set -e

# for macOS (replace 7 with version of gcc installed on your machine)
# NOTE: your gcc / g++ from Homebrew is probably in /usr/local/bin
#export CXX=/usr/local/bin/g++-8 CC=/usr/local/bin/gcc-8

# Make a new temporary folder to work in
rm -rf lightgbm_r
mkdir lightgbm_r

# copy in the relevant files
cp -R R-package/ lightgbm_r
cp -R include lightgbm_r/src/
cp -R src lightgbm_r/src/
cp CMakeLists.txt lightgbm_r/inst/bin/

# rebuild documentation
Rscript -e "devtools::document('lightgbm_r/')"

# Build the package
# NOTE: --keep-empty-dirs is necessary to keep the deep paths expected
#       by CMake while also meeting the CRAN req to create object files
#       on demand
R CMD BUILD lightgbm_r/  --keep-empty-dirs

# Install the package
VERSION=$(cat lightgbm_r/DESCRIPTION | grep Version | cut -d ' ' -f 2)
R CMD INSTALL lightgbm_${VERSION}.tar.gz --no-multiarch

# Run R CMD CHECK
#R CMD CHECK lightgbm_2.1.2.tar.gz --as-cran | tee check.log | cat
