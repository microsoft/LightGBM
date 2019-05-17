LightGBM R-package
==================

Installation
------------

### Preparation

You need to install git and [CMake](https://cmake.org/) first.

Note: 32-bit R/Rtools is not supported.

#### Windows Preparation

Installing [Rtools](https://cran.r-project.org/bin/windows/Rtools/) is mandatory, and only support the 64-bit version. It requires to add to PATH the Rtools MinGW64 folder, if it was not done automatically during installation.

The default compiler is Visual Studio (or [VS Build Tools](https://visualstudio.microsoft.com/downloads/)) in Windows, with an automatic fallback to Rtools or any [MinGW64](https://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win64/Personal%20Builds/mingw-builds/) (x86_64-posix-seh) available (this means if you have only Rtools and CMake, it will compile fine).

To force the usage of Rtools / MinGW, you can set `use_mingw` to `TRUE` in `R-package/src/install.libs.R`.

For users who wants to install online with GPU or want to choose a specific compiler, please check the end of this document for installation using a helper package ([Laurae2/lgbdl](https://github.com/Laurae2/lgbdl/)).

**Warning for Windows users**: it is recommended to use *Visual Studio* for its better multi-threading efficiency in Windows for many core systems. For very simple systems (dual core computers or worse), MinGW64 is recommended for maximum performance. If you do not know what to choose, it is recommended to use [Visual Studio](https://visualstudio.microsoft.com/downloads/), the default compiler. **Do not try using MinGW in Windows on many core systems. It may result in 10x slower results than Visual Studio.**

#### Mac OS Preparation

You can perform installation either with **Apple Clang** or **gcc**. In case you prefer **Apple Clang**, you should install **OpenMP** (details for installation can be found in [Installation Guide](https://github.com/microsoft/LightGBM/blob/master/docs/Installation-Guide.rst#apple-clang)) first and **CMake** version 3.12 or higher is required. In case you prefer **gcc**, you need to install it (details for installation can be found in [Installation Guide](https://github.com/microsoft/LightGBM/blob/master/docs/Installation-Guide.rst#gcc)) and set some environment variables to tell R to use `gcc` and `g++`. If you install these from Homebrew, your versions of `g++` and `gcc` are most likely in `/usr/local/bin`, as shown below.

```
# replace 8 with version of gcc installed on your machine
export CXX=/usr/local/bin/g++-8 CC=/usr/local/bin/gcc-8
```

### Install

Build and install R-package with the following commands:

```sh
git clone --recursive https://github.com/microsoft/LightGBM
cd LightGBM
Rscript build_r.R
```

The `build_r.R` script builds the package in a temporary directory called `lightgbm_r`. It will destroy and recreate that directory each time you run the script.

Note: for the build with Visual Studio/VS Build Tools in Windows, you should use the Windows CMD or Powershell.

Windows users may need to run with administrator rights (either R or the command prompt, depending on the way you are installing this package). Linux users might require the appropriate user write permissions for packages.

Set `use_gpu` to `TRUE` in `R-package/src/install.libs.R` to enable the build with GPU support. You will need to install Boost and OpenCL first: details for installation can be found in [Installation-Guide](https://github.com/microsoft/LightGBM/blob/master/docs/Installation-Guide.rst#build-gpu-version).

If you are using a precompiled dll/lib locally, you can move the dll/lib into LightGBM root folder, modify `LightGBM/R-package/src/install.libs.R`'s 2nd line (change `use_precompile <- FALSE` to `use_precompile <- TRUE`), and install R-package as usual. **NOTE: If your R version is not smaller than 3.5.0, you should set `DUSE_R35=ON` in cmake options when build precompiled dll/lib**.

When your package installation is done, you can check quickly if your LightGBM R-package is working by running the following:

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

* Visual Studio 2015/2017/2019 precompiled DLL: download and install Visual Studio Runtime for [2015](https://www.microsoft.com/en-us/download/details.aspx?id=52685)/[2017](https://aka.ms/vs/15/release/vc_redist.x64.exe)/[2019](https://aka.ms/vs/16/release/vc_redist.x64.exe) (you will get an error about MSVCP140.dll missing otherwise)

Once you have all this setup, you can use `lgb.dl` from `lgbdl` package to install LightGBM from repository.

For instance, you can install the R-package from LightGBM master commit of GitHub with Visual Studio using the following from R:

```r
lgb.dl(commit = "master",
       compiler = "vs",
       repo = "https://github.com/microsoft/LightGBM")
```

You may also install using a precompiled dll/lib using the following from R:

```r
lgb.dl(commit = "master",
       libdll = "C:\\LightGBM\\windows\\x64\\DLL\\lib_lightgbm.dll", # YOUR PRECOMPILED DLL
       repo = "https://github.com/microsoft/LightGBM")
```

You may also install online using a LightGBM with proper GPU support using Visual Studio (as an example here) using the following from R:

```r
lgb.dl(commit = "master",
       compiler = "vs", # Remove this for MinGW + GPU installation
       repo = "https://github.com/microsoft/LightGBM",
       use_gpu = TRUE)
```

For more details about options, please check [Laurae2/lgbdl](https://github.com/Laurae2/lgbdl/) R-package.

You may also read [Microsoft/LightGBM#912](https://github.com/microsoft/LightGBM/issues/912#issuecomment-329496254) for a visual example for LightGBM installation in Windows with Visual Studio.

Examples
--------

Please visit [demo](https://github.com/microsoft/LightGBM/tree/master/R-package/demo):

* [Basic walkthrough of wrappers](https://github.com/microsoft/LightGBM/blob/master/R-package/demo/basic_walkthrough.R)
* [Boosting from existing prediction](https://github.com/microsoft/LightGBM/blob/master/R-package/demo/boost_from_prediction.R)
* [Early Stopping](https://github.com/microsoft/LightGBM/blob/master/R-package/demo/early_stopping.R)
* [Cross Validation](https://github.com/microsoft/LightGBM/blob/master/R-package/demo/cross_validation.R)
* [Multiclass Training/Prediction](https://github.com/microsoft/LightGBM/blob/master/R-package/demo/multiclass.R)
* [Leaf (in)Stability](https://github.com/microsoft/LightGBM/blob/master/R-package/demo/leaf_stability.R)
* [Weight-Parameter Adjustment Relationship](https://github.com/microsoft/LightGBM/blob/master/R-package/demo/weight_param.R)
