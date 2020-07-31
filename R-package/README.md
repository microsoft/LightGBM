LightGBM R-package
==================

### Contents

* [Installation](#installation)
* [Examples](#examples)
* [Testing](#testing)
* [Preparing a CRAN Package and Installing It](#preparing-a-cran-package-and-installing-it)
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

If you are using a precompiled dll/lib locally, you can move the dll/lib into LightGBM root folder, modify `LightGBM/R-package/src/install.libs.R`'s 2nd line (change `use_precompile <- FALSE` to `use_precompile <- TRUE`), and install R-package as usual.

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

Preparing a CRAN Package and Installing It
------------------------------------------

This section is primarily for maintainers, but may help users and contributors to understand the structure of the R package.

Most of `LightGBM` uses `CMake` to handle tasks like setting compiler and linker flags, including header file locations, and linking to other libraries. Because CRAN packages typically do not assume the presence of `CMake`, the R package uses an alternative method that is in the CRAN-supported toolchain for building R packages with C++ code: `Autoconf`.

For more information on this approach, see ["Writing R Extensions"](https://cran.r-project.org/doc/manuals/r-release/R-exts.html#Configure-and-cleanup).

### Build a CRAN Package

From the root of the repository, run the following.

```shell
sh build-cran-package.sh
```

This will create a file `lightgbm_${VERSION}.tar.gz`, where `VERSION` is the version of `LightGBM`.

### Standard Installation from CRAN Package

After building the package, install it with a command like the following:

```shell
R CMD install lightgbm_*.tar.gz
```

#### Custom Installation (Linux, Mac)

To change the compiler used when installing the package, you can create a file `~/.R/Makevars` which overrides `CC` (`C` compiler) and `CXX` (`C++` compiler). For example, to use `gcc` instead of `clang` on Mac, you could use something like the following:

```make
# ~/.R/Makevars
CC=gcc-8
CXX=g++-8
CXX11=g++-8
```

### Changing the CRAN Package

A lot of details are handled automatically by `R CMD build` and `R CMD install`, so it can be difficult to understand how the files in the R package are related to each other. An extensive treatment of those details is available in ["Writing R Extensions"](https://cran.r-project.org/doc/manuals/r-release/R-exts.html).

This section briefly explains the key files for building a CRAN package. To update the package, edit the files relevant to your change and re-run the steps in [Build a CRAN Package](#build-a-cran-package).

**Linux or Mac**

At build time, `configure` will be run and used to create a file `Makevars`, using `Makevars.in` as a template.

1. Edit `configure.ac`
2. Create `configure` with `autoconf`. Do not edit it by hand. This file must be generated on Ubuntu 18.04.

    If you have an Ubuntu 18.04 environment available, run the provided script from the root of the `LightGBM` repository.

    ```shell
    ./R-package/recreate-configure.sh
    ```

    If you do not have easy access to an Ubuntu 18.04 environment, the `configure` script can be generated using Docker.

    ```shell
    docker run \
        -v $(pwd):/opt/LightGBM \
        -t ubuntu:18.04 \
        /bin/bash -c "cd /opt/LightGBM && ./R-package/recreate-configure.sh"
    ```

    The version of `autoconf` used by this project is stored in `R-package/AUTOCONF_UBUNTU_VERSION`. To update that version, update that file and run the commands above. To see available versions, see https://packages.ubuntu.com/search?keywords=autoconf.

3. Edit `src/Makevars.in`

**Configuring for Windows**

At build time, `configure.win` will be run and used to create a file `Makevars.win`, using `Makevars.win.in` as a template.

1. Edit `configure.win` directly
2. Edit `src/Makevars.win.in`

External (Unofficial) Repositories
----------------------------------

Projects listed here are not maintained or endorsed by the `LightGBM` development team, but may offer some features currently missing from the main R package.

* [lightgbm.py](https://github.com/kapsner/lightgbm.py): This R package offers a wrapper built with `reticulate`, a package used to call Python code from R. If you are comfortable with the added installation complexity of installing `lightgbm`'s Python package and the performance cost of passing data between R and Python, you might find that this package offers some features that are not yet available in the native `lightgbm` R package.

Known Issues
------------

For information about known issues with the R package, see the [R-package section of LightGBM's main FAQ page](https://lightgbm.readthedocs.io/en/latest/FAQ.html#r-package).
