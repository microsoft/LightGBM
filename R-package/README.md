LightGBM R Package
==================

Installation
------------

### Preparation
You need to install git and [cmake](https://cmake.org/) first.

The default compiler is Visual Studio (or [MS Build](https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017)) in Windows. You also can use Rtools (default) or [MinGW64](https://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win64/Personal%20Builds/mingw-builds/) (x86_64-posix-seh) to compile by setting `use_mingw` to `TRUE` in `R-package/src/install.libs.R`. For MinGW users who wants to install online, please check the end of this document for installation using a helper package ([Laurae2/lgbdl](https://github.com/Laurae2/lgbdl/)).

It is recommended to use *Visual Studio* for its better multi-threading efficency in Windows for many core systems. For very simple systems (dual core computers or worse), MinGW64 is recommended for maximum performance. If you do not know what to choose, it is recommended to use [Visual Studio](https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017).

For Windows users, installing [Rtools](https://cran.r-project.org/bin/windows/Rtools/) is mandatory.

For Mac OS X users, gcc with OpenMP support must be installed first. Refer to [wiki](https://github.com/Microsoft/LightGBM/wiki/Installation-Guide#osx) for installing gcc with OpenMP support.

To avoid critical package issues if you are using R 3.4.0 (not the patched/devel version), it is recommended to install once the LightGBM R package, even if it is an old version: `devtools::install_github("Microsoft/LightGBM@v1", subdir = "R-package")`. Make sure you have the correct permissions to install the package.

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

If you are using a precompiled dll/lib locally, you can move the dll/lib into LightGBM root folder, modify `LightGBM/R-package/src/install.libs.R`'s 2nd line (change `use_precompile <- FALSE` to `use_precompile <- TRUE`), and install R-package as usual.

When your package installation is done, you can check quickly if your LightGBM R package is working by running the following:

```r
library(lightgbm)
data(agaricus.train, package='lightgbm')
train <- agaricus.train
dtrain <- lgb.Dataset(train$data, label=train$label)
params <- list(objective="regression", metric="l2")
model <- lgb.cv(params, dtrain, 10, nfold=5, min_data=1, learning_rate=1, early_stopping_rounds=10)
```

Installation with precompiled dll/lib from R using GitHub
------------

You can install LightGBM R-package from GitHub with devtools thanks to a helper package for LightGBM.

### Prerequisites

You will need:

* Precompiled LightGBM dll/lib
* MinGW / Visual Studio / gcc (depending on your OS and your needs) with make in PATH environment variable
* git in PATH environment variable
* [cmake](https://cmake.org/) in PATH environment variable
* [lgbdl](https://github.com/Laurae2/lgbdl/) R-package, which can be installed using `devtools::install_github("Laurae2/lgbdl")`
* [Rtools](https://cran.r-project.org/bin/windows/Rtools/) if using Windows

In addition, if you are using a Visual Studio precompiled DLL, assuming you do not have Visual Studio installed (if you have it installed, ignore the warnings below):

* Visual Studio 2013 precompiled DLL: download and install Visual Studio Runtime for [2013](https://support.microsoft.com/en-us/help/3179560/update-for-visual-c-2013-and-visual-c-redistributable-package) (you will get an error about MSVCP120.dll missing otherwise)
* Visual Studio 2015/2017 precompiled DLL: download and install Visual Studio Runtime for [2015](https://www.microsoft.com/en-us/download/details.aspx?id=52685)/[2017](https://go.microsoft.com/fwlink/?LinkId=746572) (you will get an error about MSVCP140.dll missing otherwise)

Once you have all this setup, you can use `lgb.dl` from `lgbdl` package to install LightGBM from repository.

For instance, you can install the R package from LightGBM master commit of GitHub using the following from R:

```r
lgb.dl(commit = "master",
       compiler = "gcc",
       repo = "https://github.com/Microsoft/LightGBM",
       cores = 4)
```

You may also install using a precompiled dll/lib using the following from R:

```r
lgb.dl(commit = "master",
       libdll = "C:\\LightGBM\\windows\\x64\\DLL\\lib_lightgbm.dll", # YOUR PRECOMPILED DLL
       repo = "https://github.com/Microsoft/LightGBM")
```

For more details about options, please check [Laurae2/lgbdl](https://github.com/Laurae2/lgbdl/) R-package.

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
