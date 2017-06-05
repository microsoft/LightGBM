LightGBM R Package
==================

Installation
------------

### Preparation
You need to install git and [cmake](https://cmake.org/) first.

The default compiler is Visual Studio (or [MS Build](https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017)) in Windows. You also can use Rtools (default) or [MinGW64](https://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win64/Personal%20Builds/mingw-builds/) (x86_64-posix-seh) to compile by setting `use_mingw` to `TRUE` in `R-package/src/install.libs.R`.

It is recommended to use *Visual Studio* for its better multi-threading efficency in Windows for many core systems. For simple systems (like laptops or small desktops), MinGW64 is recommended.

For Mac OS X users, gcc with OpenMP support must be installed first. Refer to [wiki](https://github.com/Microsoft/LightGBM/wiki/Installation-Guide#osx) for installing gcc with OpenMP support.

### Install
Install LightGBM R-package with the following command:

```sh
git clone --recursive https://github.com/Microsoft/LightGBM
cd LightGBM/R-package
R CMD INSTALL --build .
```

Or build a self-contained R package which can be installed afterwards:

```sh
git clone --recursive https://github.com/Microsoft/LightGBM
cd LightGBM/R-package
Rscript build_package.R
R CMD INSTALL lightgbm_0.2.tar.gz
``` 

Windows users may need to run with administrator rights (either R or the command prompt, depending on the way you are installing this package). Rtools must be installed for Windows. Linux users might require the appropriate user write permissions for packages.

Set `use_gpu` to `TRUE` in `R-package/src/install.libs.R` to enable the build with GPU support. You will need to install Boost and OpenCL first: details for installation can be found in [gpu-support](https://github.com/Microsoft/LightGBM/wiki/Installation-Guide#with-gpu-support).

You can also install directly from R using the repository with `devtools`:

```r
devtools::install_github("Microsoft/LightGBM", subdir = "R-package")
```

When your package installation is done, you can check quickly if your LightGBM R package is working by running the following:

```r
library(lightgbm)
data(agaricus.train, package='lightgbm')
train <- agaricus.train
dtrain <- lgb.Dataset(train$data, label=train$label)
params <- list(objective="regression", metric="l2")
model <- lgb.cv(params, dtrain, 10, nfold=5, min_data=1, learning_rate=1, early_stopping_rounds=10)
```

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
