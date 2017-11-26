LightGBM R Package
==================

Installation
------------

### Preparation

You need to install git and [CMake](https://cmake.org/) first.

Note: 32-bit R/Rtools is not supported.

#### Windows Preparation

Installing [Rtools](https://cran.r-project.org/bin/windows/Rtools/) is mandatory, and only support the 64-bit version. It requires to add to PATH the Rtools MinGW64 folder, if it was not done automatically during installation.

The default compiler is Visual Studio (or [MS Build](https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017)) in Windows, with an automatic fallback to Rtools or any [MinGW64](https://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win64/Personal%20Builds/mingw-builds/) (x86_64-posix-seh) available (this means if you have only Rtools and CMake, it will compile fine).

To force the usage of Rtools / MinGW, you can set `use_mingw` to `TRUE` in `R-package/src/install.libs.R`.

For users who wants to install online with GPU or want to choose a specific compiler, please check the end of this document for installation using a helper package ([Laurae2/lgbdl](https://github.com/Laurae2/lgbdl/)).

**Warning for Windows users**: it is recommended to use *Visual Studio* for its better multi-threading efficency in Windows for many core systems. For very simple systems (dual core computers or worse), MinGW64 is recommended for maximum performance. If you do not know what to choose, it is recommended to use [Visual Studio](https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017), the default compiler. **Do not try using MinGW in Windows on many core systems. It may result in 10x slower results than Visual Studio.**

#### macOS Preparation

gcc with OpenMP support must be installed first. Refer to [Installation-Guide](https://github.com/Microsoft/LightGBM/blob/master/docs/Installation-Guide.rst#macos) for installing gcc with OpenMP support.

### Install

Install LightGBM R-package with the following command:

```sh
git clone --recursive https://github.com/Microsoft/LightGBM
cd LightGBM/R-package
# export CXX=g++-7 CC=gcc-7 # for macOS
R CMD INSTALL --build . --no-multiarch
```

Or build a self-contained R package which can be installed afterwards:

```sh
git clone --recursive https://github.com/Microsoft/LightGBM
cd LightGBM/R-package
Rscript build_package.R
# export CXX=g++-7 CC=gcc-7 # for macOS
R CMD INSTALL lightgbm_2.0.4.tar.gz --no-multiarch
``` 

Note: for the build with Visual Studio/MSBuild in Windows, you should use the Windows CMD or Powershell.

Windows users may need to run with administrator rights (either R or the command prompt, depending on the way you are installing this package). Linux users might require the appropriate user write permissions for packages.

Set `use_gpu` to `TRUE` in `R-package/src/install.libs.R` to enable the build with GPU support. You will need to install Boost and OpenCL first: details for installation can be found in [Installation-Guide](https://github.com/Microsoft/LightGBM/blob/master/docs/Installation-Guide.rst#build-gpu-version).

You can also install directly from R using the repository with `devtools`:

```r
library(devtools)
options(devtools.install.args = "--no-multiarch") # if you have 64-bit R only, you can skip this
install_github("Microsoft/LightGBM", subdir = "R-package")
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

Installation with Precompiled dll/lib from R Using GitHub
---------------------------------------------------------

You can install LightGBM R-package from GitHub with devtools thanks to a helper package for LightGBM.

### Prerequisites

You will need:

* Precompiled LightGBM dll/lib
* MinGW / Visual Studio / gcc (depending on your OS and your needs) with make in PATH environment variable
* git in PATH environment variable
* [CMake](https://cmake.org/) in PATH environment variable
* [lgbdl](https://github.com/Laurae2/lgbdl/) R-package, which can be installed using `devtools::install_github("Laurae2/lgbdl")`
* [Rtools](https://cran.r-project.org/bin/windows/Rtools/) if using Windows

In addition, if you are using a Visual Studio precompiled DLL, assuming you do not have Visual Studio installed (if you have it installed, ignore the warnings below):

* Visual Studio 2015/2017 precompiled DLL: download and install Visual Studio Runtime for [2015](https://www.microsoft.com/en-us/download/details.aspx?id=52685)/[2017](https://go.microsoft.com/fwlink/?LinkId=746572) (you will get an error about MSVCP140.dll missing otherwise)

Once you have all this setup, you can use `lgb.dl` from `lgbdl` package to install LightGBM from repository.

For instance, you can install the R package from LightGBM master commit of GitHub with Visual Studio using the following from R:

```r
lgb.dl(commit = "master",
       compiler = "vs",
       repo = "https://github.com/Microsoft/LightGBM")
```

You may also install using a precompiled dll/lib using the following from R:

```r
lgb.dl(commit = "master",
       libdll = "C:\\LightGBM\\windows\\x64\\DLL\\lib_lightgbm.dll", # YOUR PRECOMPILED DLL
       repo = "https://github.com/Microsoft/LightGBM")
```

You may also install online using a LightGBM with proper GPU support using Visual Studio (as an example here) using the following from R:

```r
lgb.dl(commit = "master",
       compiler = "vs", # Remove this for MinGW + GPU installation
       repo = "https://github.com/Microsoft/LightGBM",
       use_gpu = TRUE)
```

For more details about options, please check [Laurae2/lgbdl](https://github.com/Laurae2/lgbdl/) R-package.

You may also read [Microsoft/LightGBM#912](https://github.com/Microsoft/LightGBM/issues/912#issuecomment-329496254) for a visual example for LightGBM installation in Windows with Visual Studio.

Examples
--------

Please visit [demo](demo):

* [Basic walkthrough of wrappers](demo/basic_walkthrough.R)
* [Boosting from existing prediction](demo/boost_from_prediction.R)
* [Early Stopping](demo/early_stopping.R)
* [Cross Validation](demo/cross_validation.R)
* [Multiclass Training/Prediction](demo/multiclass.R)
* [Leaf (in)Stability](demo/leaf_stability.R)
* [Weight-Parameter Adjustment Relationship](demo/weight_param.R)
