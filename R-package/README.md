# LightGBM R-package

[![CRAN Version](https://www.r-pkg.org/badges/version/lightgbm)](https://cran.r-project.org/package=lightgbm)
[![Downloads](https://cranlogs.r-pkg.org/badges/grand-total/lightgbm)](https://cran.r-project.org/package=lightgbm)
[![API Docs](https://readthedocs.org/projects/lightgbm/badge/?version=latest)](https://lightgbm.readthedocs.io/en/latest/R/reference/)

<img src="man/figures/logo.svg" align="right" alt="" width="175" />

### Contents

* [Installation](#installation)
    - [Installing the CRAN Package](#installing-the-cran-package)
    - [Installing from Source with CMake](#install)
    - [Installing a GPU-enabled Build](#installing-a-gpu-enabled-build)
    - [Installing Precompiled Binaries](#installing-precompiled-binaries)
    - [Installing from a Pre-compiled lib_lightgbm](#lib_lightgbm)
* [Examples](#examples)
* [Testing](#testing)
* [Preparing a CRAN Package](#preparing-a-cran-package)
* [External Repositories](#external-unofficial-repositories)
* [Known Issues](#known-issues)

Installation
------------

For the easiest installation, go to ["Installing the CRAN package"](#installing-the-cran-package).

If you experience any issues with that, try ["Installing from Source with CMake"](#install). This can produce a more efficient version of the library on Windows systems with Visual Studio.

To build a GPU-enabled version of the package, follow the steps in ["Installing a GPU-enabled Build"](#installing-a-gpu-enabled-build).

If any of the above options do not work for you or do not meet your needs, please let the maintainers know by [opening an issue](https://github.com/microsoft/LightGBM/issues).

When your package installation is done, you can check quickly if your LightGBM R-package is working by running the following:

```r
library(lightgbm)
data(agaricus.train, package='lightgbm')
train <- agaricus.train
dtrain <- lgb.Dataset(train$data, label = train$label)
model <- lgb.cv(
    params = list(
        objective = "regression"
        , metric = "l2"
    )
    , data = dtrain
)
```

### Installing the CRAN package

`{lightgbm}` is [available on CRAN](https://cran.r-project.org/package=lightgbm), and can be installed with the following R code.

```r
install.packages("lightgbm", repos = "https://cran.r-project.org")
```

This is the easiest way to install `{lightgbm}`. It does not require `CMake` or `Visual Studio`, and should work well on many different operating systems and compilers.

Each CRAN package is also available on [LightGBM releases](https://github.com/microsoft/LightGBM/releases), with a name like `lightgbm-{VERSION}-r-cran.tar.gz`.

#### Custom Installation (Linux, Mac)

The steps above should work on most systems, but users with highly-customized environments might want to change how R builds packages from source.

To change the compiler used when installing the CRAN package, you can create a file `~/.R/Makevars` which overrides `CC` (`C` compiler) and `CXX` (`C++` compiler).

For example, to use `gcc` instead of `clang` on Mac, you could use something like the following:

```make
# ~/.R/Makevars
CC=gcc-8
CXX=g++-8
CXX11=g++-8
```

### Installing from Source with CMake <a name="install"></a>

You need to install git and [CMake](https://cmake.org/) first.

Note: this method is only supported on 64-bit systems. If you need to run LightGBM on 32-bit Windows (i386), follow the instructions in ["Installing the CRAN Package"](#installing-the-cran-package).

#### Windows Preparation

NOTE: Windows users may need to run with administrator rights (either R or the command prompt, depending on the way you are installing this package).

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

If you want to force `LightGBM` to use MinGW (for any R version), pass `--use-mingw` to the installation script.

```shell
Rscript build_r.R --use-mingw
```

**MSYS2 (R 4.x)**

If you are using R 4.x and installation fails with Visual Studio, `LightGBM` will fall back to using [MSYS2](https://www.msys2.org/). This should work with the tools already bundled in `Rtools` 4.0.

If you want to force `LightGBM` to use MSYS2 (for any R version), pass `--use-msys2` to the installation script.

```shell
Rscript build_r.R --use-msys2
```

#### Mac OS Preparation

You can perform installation either with **Apple Clang** or **gcc**. In case you prefer **Apple Clang**, you should install **OpenMP** (details for installation can be found in [Installation Guide](https://github.com/microsoft/LightGBM/blob/master/docs/Installation-Guide.rst#apple-clang)) first and **CMake** version 3.16 or higher is required. In case you prefer **gcc**, you need to install it (details for installation can be found in [Installation Guide](https://github.com/microsoft/LightGBM/blob/master/docs/Installation-Guide.rst#gcc)) and set some environment variables to tell R to use `gcc` and `g++`. If you install these from Homebrew, your versions of `g++` and `gcc` are most likely in `/usr/local/bin`, as shown below.

```
# replace 8 with version of gcc installed on your machine
export CXX=/usr/local/bin/g++-8 CC=/usr/local/bin/gcc-8
```

#### Install with CMake

After following the "preparation" steps above for your operating system, build and install the R-package with the following commands:

```sh
git clone --recursive https://github.com/microsoft/LightGBM
cd LightGBM
Rscript build_r.R
```

The `build_r.R` script builds the package in a temporary directory called `lightgbm_r`. It will destroy and recreate that directory each time you run the script. That script supports the following command-line options:

- `--skip-install`: Build the package tarball, but do not install it.
- `--use-gpu`: Build a GPU-enabled version of the library.
- `--use-mingw`: Force the use of MinGW toolchain, regardless of R version.
- `--use-msys2`: Force the use of MSYS2 toolchain, regardless of R version.

Note: for the build with Visual Studio/VS Build Tools in Windows, you should use the Windows CMD or PowerShell.

### Installing a GPU-enabled Build

You will need to install Boost and OpenCL first: details for installation can be found in [Installation-Guide](https://github.com/microsoft/LightGBM/blob/master/docs/Installation-Guide.rst#build-gpu-version).

After installing these other libraries, follow the steps in ["Installing from Source with CMake"](#install). When you reach the step that mentions `build_r.R`, pass the flag `--use-gpu`.

```shell
Rscript build_r.R --use-gpu
```

You may also need or want to provide additional configuration, depending on your setup. For example, you may need to provide locations for Boost and OpenCL.

```shell
Rscript build_r.R \
    --use-gpu \
    --opencl-library=/usr/lib/x86_64-linux-gnu/libOpenCL.so \
    --boost-librarydir=/usr/lib/x86_64-linux-gnu
```

The following options correspond to the [CMake FindBoost options](https://cmake.org/cmake/help/latest/module/FindBoost.html) by the same names.

* `--boost-root`
* `--boost-dir`
* `--boost-include-dir`
* `--boost-librarydir`

The following options correspond to the [CMake FindOpenCL options](https://cmake.org/cmake/help/latest/module/FindOpenCL.html) by the same names.

* `--opencl-include-dir`
* `--opencl-library`

### Installing Precompiled Binaries

Precompiled binaries for Mac and Windows are prepared by CRAN a few days after each release to CRAN. They can be installed with the following R code.

```r
install.packages(
    "lightgbm"
    , type = "both"
    , repos = "https://cran.r-project.org"
)
```

These packages do not require compilation, so they will be faster and easier to install than packages that are built from source.

CRAN does not prepare precompiled binaries for Linux, and as of this writing neither does this project.

### Installing from a Pre-compiled lib_lightgbm <a name="lib_lightgbm"></a>

Previous versions of LightGBM offered the ability to first compile the C++ library (`lib_lightgbm.so` or `lib_lightgbm.dll`) and then build an R package that wraps it.

As of version 3.0.0, this is no longer supported. If building from source is difficult for you, please [open an issue](https://github.com/microsoft/LightGBM/issues).

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

The R package's unit tests are run automatically on every commit, via integrations like [GitHub Actions](https://github.com/microsoft/LightGBM/actions). Adding new tests in `R-package/tests/testthat` is a valuable way to improve the reliability of the R package.

When adding tests, you may want to use test coverage to identify untested areas and to check if the tests you've added are covering all branches of the intended code.

The example below shows how to generate code coverage for the R package on a macOS or Linux setup. To adjust for your environment, refer to [the customization step described above](#custom-installation-linux-mac).

```shell
# Install
sh build-cran-package.sh

# Get coverage
Rscript -e " \
    coverage  <- covr::package_coverage('./lightgbm_r', type = 'tests', quiet = FALSE);
    print(coverage);
    covr::report(coverage, file = file.path(getwd(), 'coverage.html'), browse = TRUE);
    "
```

Preparing a CRAN Package
------------------------

This section is primarily for maintainers, but may help users and contributors to understand the structure of the R package.

Most of `LightGBM` uses `CMake` to handle tasks like setting compiler and linker flags, including header file locations, and linking to other libraries. Because CRAN packages typically do not assume the presence of `CMake`, the R package uses an alternative method that is in the CRAN-supported toolchain for building R packages with C++ code: `Autoconf`.

For more information on this approach, see ["Writing R Extensions"](https://cran.r-project.org/doc/manuals/r-release/R-exts.html#Configure-and-cleanup).

### Build a CRAN Package

From the root of the repository, run the following.

```shell
git submodule update --init --recursive
sh build-cran-package.sh
```

This will create a file `lightgbm_${VERSION}.tar.gz`, where `VERSION` is the version of `LightGBM`.

Also, CRAN package is generated with every commit to any repo's branch and can be found in "Artifacts" section of the associated Azure Pipelines run.

### Standard Installation from CRAN Package

After building the package, install it with a command like the following:

```shell
R CMD install lightgbm_*.tar.gz
```

### Changing the CRAN Package

A lot of details are handled automatically by `R CMD build` and `R CMD install`, so it can be difficult to understand how the files in the R package are related to each other. An extensive treatment of those details is available in ["Writing R Extensions"](https://cran.r-project.org/doc/manuals/r-release/R-exts.html).

This section briefly explains the key files for building a CRAN package. To update the package, edit the files relevant to your change and re-run the steps in [Build a CRAN Package](#build-a-cran-package).

**Linux or Mac**

At build time, `configure` will be run and used to create a file `Makevars`, using `Makevars.in` as a template.

1. Edit `configure.ac`.
2. Create `configure` with `autoconf`. Do not edit it by hand. This file must be generated on Ubuntu 20.04.

    If you have an Ubuntu 20.04 environment available, run the provided script from the root of the `LightGBM` repository.

    ```shell
    ./R-package/recreate-configure.sh
    ```

    If you do not have easy access to an Ubuntu 20.04 environment, the `configure` script can be generated using Docker by running the code below from the root of this repo.

    ```shell
    docker run \
        -v $(pwd):/opt/LightGBM \
        -w /opt/LightGBM \
        -t ubuntu:20.04 \
        ./R-package/recreate-configure.sh
    ```

    The version of `autoconf` used by this project is stored in `R-package/AUTOCONF_UBUNTU_VERSION`. To update that version, update that file and run the commands above. To see available versions, see https://packages.ubuntu.com/search?keywords=autoconf.

3. Edit `src/Makevars.in`.

Alternatively, GitHub Actions can re-generate this file for you. On a pull request (only on internal one, does not work for ones from forks), create a comment with this phrase:

> /gha run r-configure

**Configuring for Windows**

At build time, `configure.win` will be run and used to create a file `Makevars.win`, using `Makevars.win.in` as a template.

1. Edit `configure.win` directly.
2. Edit `src/Makevars.win.in`.

### Testing the CRAN Package

`{lightgbm}` is tested automatically on every commit, across many combinations of operating system, R version, and compiler. This section describes how to test the package locally while you are developing.

#### Windows, Mac, and Linux

```shell
sh build-cran-package.sh
R CMD check --as-cran lightgbm_*.tar.gz
```

#### Solaris

All packages uploaded to CRAN must pass `R CMD check` on Solaris 10. To test LightGBM on this operating system, you can use the free service [R Hub](https://builder.r-hub.io/), a free service generously provided by the R Consortium.

```shell
sh build-cran-package.sh
```

```r
package_tarball <- paste0("lightgbm_", readLines("VERSION.txt")[1], ".tar.gz")
rhub::check(
    path = package_tarball
    , email = "your_email_here"
    , check_args = "--as-cran"
    , platform = c(
        "solaris-x86-patched"
        , "solaris-x86-patched-ods"
    )
    , env_vars = c(
        "R_COMPILE_AND_INSTALL_PACKAGES" = "always"
    )
)
```

Alternatively, GitHub Actions can run code above for you. On a pull request, create a comment with this phrase:

> /gha run r-solaris

**NOTE:** Please do this only once you see that other R tests on a pull request are passing. R Hub is a free resource with limited capacity, and we want to be respectful community members.

#### UBSAN

All packages uploaded to CRAN must pass a build using `gcc` instrumented with two sanitizers: the Address Sanitizer (ASAN) and the Undefined Behavior Sanitizer (UBSAN). For more background, see [this blog post](http://dirk.eddelbuettel.com/code/sanitizers.html).

You can replicate these checks locally using Docker.

```shell
docker run \
    -v $(pwd):/opt/LightGBM \
    -w /opt/LightGBM \
    -it rhub/rocker-gcc-san \
    /bin/bash

Rscript -e "install.packages(c('R6', 'data.table', 'jsonlite', 'testthat'), repos = 'https://cran.rstudio.com', Ncpus = parallel::detectCores())"

sh build-cran-package.sh

Rdevel CMD install lightgbm_*.tar.gz
cd R-package/tests
Rscriptdevel testthat.R
```

#### Valgrind

All packages uploaded to CRAN must be built and tested without raising any issues from `valgrind`. `valgrind` is a profiler that can catch serious issues like memory leaks and illegal writes. For more information, see [this blog post](https://reside-ic.github.io/blog/debugging-and-fixing-crans-additional-checks-errors/).

You can replicate these checks locally using Docker. Note that instrumented versions of R built to use `valgrind` run much slower, and these tests may take as long as 20 minutes to run.

```shell
docker run \
    -v $(pwd):/opt/LightGBM \
    -w /opt/LightGBM \
    -it \
        wch1/r-debug

RDscriptvalgrind -e "install.packages(c('R6', 'data.table', 'jsonlite', 'testthat'), repos = 'https://cran.rstudio.com', Ncpus = parallel::detectCores())"

sh build-cran-package.sh

RDvalgrind CMD INSTALL \
    --preclean \
    --install-tests \
        lightgbm_*.tar.gz

cd R-package/tests

RDvalgrind \
    --no-readline \
    --vanilla \
    -d "valgrind --tool=memcheck --leak-check=full --track-origins=yes" \
        -f testthat.R \
2>&1 \
| tee out.log \
| cat
```

These tests can also be triggered on any pull request by leaving a comment in a pull request:

> /gha run r-valgrind

External (Unofficial) Repositories
----------------------------------

Projects listed here are not maintained or endorsed by the `LightGBM` development team, but may offer some features currently missing from the main R package.

* [lightgbm.py](https://github.com/kapsner/lightgbm.py): This R package offers a wrapper built with `reticulate`, a package used to call Python code from R. If you are comfortable with the added installation complexity of installing `lightgbm`'s Python package and the performance cost of passing data between R and Python, you might find that this package offers some features that are not yet available in the native `lightgbm` R package.

Known Issues
------------

For information about known issues with the R package, see the [R-package section of LightGBM's main FAQ page](https://lightgbm.readthedocs.io/en/latest/FAQ.html#r-package).
