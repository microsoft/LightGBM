LightGBM R Package
==================

Installation
------------

Windows users may need to run with administrator rights (either R or the command prompt, depending on the way you are installing this package). Rtools must be installed for Windows. Linux users might require the appropriate user write permissions for packages.

1. Following [Installation Guide](https://github.com/Microsoft/LightGBM/wiki/Installation-Guide) to build LightGBM first.
   For the windows user, please change the build config to ``DLL``.

2. Install by following command:

```
cd R-package
R CMD INSTALL --build  .
``` 

Note: build by ```devtools::install_github``` is not supportted any more.


If you want to build the self-contained R package:

1. Copy ```lib_lightgbm.so``` (or ```lib_lightgbm.dll``` in Windows) to ```R-package``` folder.

2. Build package via ```R CMD build --no-build-vignettes --binary .```

3. Install via ```R CMD INSTALL lightgbm_0.1.tar.gz```


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

Note:

* For `LDFLAGS=-L/usr/local/Cellar/gcc/6.3.0/lib` and `CPPFLAGS=-I/usr/local/Cellar/gcc/6.3.0/include`, you may need to change `6.3.0` to your gcc version.
* For `gcc-6` and `g++-6`, you may need to change to your gcc version (like `gcc-7` and `g++7` if using gcc with version 7).
* For `CXX1X`, if you are using R 3.4 or a more recent version, you must change it to `CXX11`.

To check your LightGBM installation, the test is identical to Linux/Windows versions (check the test provided just before OSX Installation part)

Performance note
------------

With `gcc`, it is recommended to use `-O3 -mtune=native` instead of the default `-O2 -mtune=core2` by modifying the appropriate file (`Makeconf` or `Makevars`) if you want to achieve maximum speed.

Benchmark example using Intel Ivy Bridge CPU on 1M x 1K dataset:

| Compilation Flag | Performance Index |
| --- | ---: |
| `-O2 -mtune=core2` | 100.00% |
| `-O2 -mtune=native` | 100.90% |
| `-O3 -mtune=native` | 102.78% |
| `-O3 -ffast-math -mtune=native` | 100.64% |

Examples
------------

Please visit [demo](demo):

* [Basic walkthrough of wrappers](demo/basic_walkthrough.R)
* [Boosting from existing prediction](demo/boost_from_prediction.R)
* [Early Stopping](demo/early_stopping.R)
* [Cross Validation](demo/cross_validation.R)
* [Multiclass Training/Prediction](demo/multiclass.R)
* [Leaf (in)Stability](demo/leaf_stability.R)
* [Weight-Parameter Adjustment Relationship](demo/weight_param.R)
