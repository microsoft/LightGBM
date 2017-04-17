LightGBM R Package
==================

Installation
------------

Windows users may need to run with administrator rights (either R or the command prompt, depending on the way you are installing this package). Rtools must be installed for Windows. Linux users might require the appropriate user write permissions for packages.

You can use a command prompt to install via command line:

```
cd R-package
R CMD INSTALL --build  .
```

You can also install directly from R using the repository with `devtools`:

```r
devtools::install_github("Microsoft/LightGBM", subdir = "R-package")
```

For the `devtools` install scenario, you can safely ignore this message:

```r
Warning message:
GitHub repo contains submodules, may not function as expected! 
```

If you want to build the self-contained R package, you can run ```unix_build_package.sh```(for UNIX) or ```win_build_package.cmd ```(for Windows). Then use ```R CMD INSTALL lightgbm_0.1.tar.gz``` to install.

When your package installation is done, you can check quickly if your LightGBM R package is working by running the following:

```r
library(lightgbm)
data(agaricus.train, package='lightgbm')
train <- agaricus.train
dtrain <- lgb.Dataset(train$data, label=train$label)
params <- list(objective="regression", metric="l2")
model <- lgb.cv(params, dtrain, 10, nfold=5, min_data=1, learning_rate=1, early_stopping_rounds=10)
```
### OSX installation 

The default installation cannot successfully complete in OSX because clang doesn't support OpenMP.

You can use the following script to change default compiler to gcc, then compile LightGBM R package:

```bash
brew install gcc --without-multilib
mkdir -p ~/.R
touch ~/.R/Makevars
cat <<EOF >>~/.R/Makevars
C=gcc-6
CXX=g++-6
CXX1X=g++-6
LDFLAGS=-L/usr/local/Cellar/gcc/6.3.0/lib
CPPFLAGS=-I/usr/local/Cellar/gcc/6.3.0/include
SHLIB_OPENMP_CFLAGS = -fopenmp
SHLIB_OPENMP_CXXFLAGS = -fopenmp
SHLIB_OPENMP_FCFLAGS = -fopenmp
SHLIB_OPENMP_FFLAGS = -fopenmp
EOF
```

Note: for `LDFLAGS=-L/usr/local/Cellar/gcc/6.3.0/lib` and `CPPFLAGS=-I/usr/local/Cellar/gcc/6.3.0/include`, you may need to change `6.3.0` to your gcc version.

To check your LightGBM installation, the test is identical to Linux/Windows versions (check the test provided just before OSX Installation part)

Examples
------------

* Please visit [demo](demo).
