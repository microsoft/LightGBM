LightGBM R Package
==================

Installation
------------

Windows users may need to run with administrator rights (either R or the command prompt, depending on the way you are installing this package). Rtools must be installed.

You can use a command prompt to install via command line:

```
cd R-package
R CMD INSTALL --build  .
```

You can also install directly from R using the repository with `devtools`:

```r
devtools::install_github("Microsoft/LightGBM", subdir = "R-package")
```


You can check quickly if your LightGBM R package is working by running the following:

```r
library(lightgbm)
data(agaricus.train, package='lightgbm')
train <- agaricus.train
dtrain <- lgb.Dataset(train$data, label=train$label)
params <- list(objective="regression", metric="l2")
model <- lgb.cv(params, dtrain, 10, nfold=5, min_data=1, learning_rate=1, early_stopping_rounds=10)
```
### OSX installation 

The default installation cannot successfully in OSX due to clang in OSX doesn't support openmp.
You can use following scirpts to change default compiler to gcc, then complie LightGBM R-package:
```
brew install gcc --without-multilib
touch ~/.R/Makevars
cat <<EOF >~/.R/Makevars
C=gcc-6
CXX=g++-6
CXX1X=g++-6
SHLIB_OPENMP_CFLAGS = -fopenmp
SHLIB_OPENMP_CXXFLAGS = -fopenmp
SHLIB_OPENMP_FCFLAGS = -fopenmp
SHLIB_OPENMP_FFLAGS = -fopenmp
EOF 
```


Examples
------------

* Please visit [demo](demo).
