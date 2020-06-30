LightGBM R-package
==================

### Contents

* [Installation](#installation)
* [Examples](#examples)
* [Testing](#testing)
* [External Repositories](#external-unofficial-repositories)
* [Known Issues](#known-issues)

Installation
------------

### Preparation

You need to install git and [CMake](https://cmake.org/) first.

Note: 32-bit (i386) R/Rtools is currently not supported.

#### Windows Preparation

Installing a 64-bit version of [Rtools](https://cran.r-project.org/bin/windows/Rtools/) is mandatory.

After installing `Rtools` and `CMake`, be sure the following paths are added to the environment variable `PATH`. These may have been automatically added when installing other software.

* `Rtools`
    - If you have `Rtools` 3.x, example:
        - `C:\Rtools\mingw_64\bin`
    - If you have `Rtools` 4.0, example:
        - `C:\rtools40\mingw64\bin`
        - `C:\rtools40\usr\bin`
* `CMake`
    - example: `C:\Program Files\CMake\bin`
* `R`
    - example: `C:\Program Files\R\R-3.6.1\bin`

NOTE: Two `Rtools` paths are required from `Rtools` 4.0 onwards because paths and the list of included software was changed in `Rtools` 4.0.

#### Windows Toolchain Options

A "toolchain" refers to the collection of software used to build the library. The R package can be built with three different toolchains.

**Warning for Windows users**: it is recommended to use *Visual Studio* for its better multi-threading efficiency in Windows for many core systems. For very simple systems (dual core computers or worse), MinGW64 is recommended for maximum performance. If you do not know what to choose, it is recommended to use [Visual Studio](https://visualstudio.microsoft.com/downloads/), the default compiler. **Do not try using MinGW in Windows on many core systems. It may result in 10x slower results than Visual Studio.**

**Visual Studio (default)**

By default, the package will be built with [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/).

**MinGW (R 3.x)**

If you are using R 3.x and installation fails with Visual Studio, `LightGBM` will fall back to using [MinGW](http://mingw-w64.org/doku.php) bundled with `Rtools`.

If you want to force `LightGBM` to use MinGW (for any R version), open `R-package/src/install.libs.R` and change `use_mingw`:

```r
use_mingw <- TRUE
```

**MSYS2 (R 4.x)**

If you are using R 4.x and installation fails with Visual Studio, `LightGBM` will fall back to using [MSYS2](https://www.msys2.org/). This should work with the tools already bundled in `Rtools` 4.0.

If you want to force `LightGBM` to use MSYS2 (for any R version), open `R-package/src/install.libs.R` and change `use_msys2`:

```r
use_msys2 <- TRUE
```

#### Mac OS Preparation

You can perform installation either with **Apple Clang** or **gcc**. In case you prefer **Apple Clang**, you should install **OpenMP** (details for installation can be found in [Installation Guide](https://github.com/microsoft/LightGBM/blob/master/docs/Installation-Guide.rst#apple-clang)) first and **CMake** version 3.16 or higher is required. In case you prefer **gcc**, you need to install it (details for installation can be found in [Installation Guide](https://github.com/microsoft/LightGBM/blob/master/docs/Installation-Guide.rst#gcc)) and set some environment variables to tell R to use `gcc` and `g++`. If you install these from Homebrew, your versions of `g++` and `gcc` are most likely in `/usr/local/bin`, as shown below.

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

Testing
-------

The R package's unit tests are run automatically on every commit, via integrations like [Travis CI](https://travis-ci.org/microsoft/LightGBM/) and [Azure DevOps](https://dev.azure.com/lightgbm-ci/lightgbm-ci/_build). Adding new tests in `R-package/tests/testthat` is a valuable way to improve the reliability of the R package.

When adding tests, you may want to use test coverage to identify untested areas and to check if the tests you've added are covering all branches of the intended code.

The example below shows how to generate code coverage for the R package on a macOS or Linux setup, using `gcc-8` to compile `LightGBM`. To adjust for your environment, swap out the 'Install' step with [the relevant code from the instructions above](#install).

```shell
# Install
export CXX=/usr/local/bin/g++-8
export CC=/usr/local/bin/gcc-8
Rscript build_r.R --skip-install

# Get coverage
Rscript -e " \
    coverage  <- covr::package_coverage('./lightgbm_r', quiet=FALSE);
    print(coverage);
    covr::report(coverage, file = file.path(getwd(), 'coverage.html'), browse = TRUE);
    "
```

External (Unofficial) Repositories
----------------------------------

Projects listed here are not maintained or endorsed by the `LightGBM` development team, but may offer some features currently missing from the main R package.

* [lightgbm.py](https://github.com/kapsner/lightgbm.py): This R package offers a wrapper built with `reticulate`, a package used to call Python code from R. If you are comfortable with the added installation complexity of installing `lightgbm`'s Python package and the performance cost of passing data between R and Python, you might find that this package offers some features that are not yet available in the native `lightgbm` R package.

Known Issues
------------

For information about known issues with the R package, see the [R-package section of LightGBM's main FAQ page](https://lightgbm.readthedocs.io/en/latest/FAQ.html#r-package).
