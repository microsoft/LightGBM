# LightGBM R-package

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

As of this writing, `LightGBM`'s R package is not available on CRAN. However, start with `LightGBM` 3.0.0, you can install a released source distribution. This is the same type of package that you'd install from CRAN. It does not require `CMake`, `Visual Studio`, or anything else outside the CRAN toolchain.

To install this package on any operating system:

1. Choose a release from [the "Releases" page](https://github.com/microsoft/LightGBM/releases).
2. Look for the artifact with a name like `lightgbm-{VERSION}-r-cran.tar.gz`. Right-click it and choose "copy link address".
3. Copy that link into `PKG_URL` in the code below and run it.

```r
PKG_URL <- "https://github.com/microsoft/LightGBM/releases/download/v3.0.0/lightgbm-3.0.0-r-cran.tar.gz"

remotes::install_url(PKG_URL)
```

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

#### Install with CMake

After following the "preparation" steps above for your operating system, build and install the R-package with the following commands:

```sh
git clone --recursive https://github.com/microsoft/LightGBM
cd LightGBM
Rscript build_r.R
```

The `build_r.R` script builds the package in a temporary directory called `lightgbm_r`. It will destroy and recreate that directory each time you run the script.

Note: for the build with Visual Studio/VS Build Tools in Windows, you should use the Windows CMD or Powershell.

### Installing a GPU-enabled Build

Set `use_gpu` to `TRUE` in `R-package/src/install.libs.R` to enable the build with GPU support. You will need to install Boost and OpenCL first: details for installation can be found in [Installation-Guide](https://github.com/microsoft/LightGBM/blob/master/docs/Installation-Guide.rst#build-gpu-version).

After installing these other libraries, follow the steps in ["Installing from Source with CMake"](#install).

### Installing Precompiled Binaries

**NOTE:** As of this writing, the precompiled binaries of the R package should be considered experimental. If you try them an experience any problems, please [open an issue](https://github.com/microsoft/LightGBM/issues).

Starting with `LightGBM` 3.0.0, precompiled binaries for the R package are created for each release. These packages do not require compilation, so they will be faster and easier to install than packages that are built from source. These packages are created with R 4.0 and are not guaranteed to work with other R versions.

Binaries are available for Windows, Mac, and Linux systems. They are not guaranteed to work with all variants and versions of these operating systems. Please [open an issue](https://github.com/microsoft/LightGBM/issues) if you encounter any problems.

To install a binary for the R package:

1. Choose a release from [the "Releases" page](https://github.com/microsoft/LightGBM/releases).
2. Choose a file based on your operating system. Right-click it and choose "copy link address".
    * Linux: `lightgbm-{VERSION}-r40-linux.tgz`
    * Mac: `lightgbm-{VERSION}-r40-macos.tgz`
    * Windows: `lightgbm-{VERSION}-r40-windows.zip`
3. Copy that link into `PKG_URL` in the code below and run it.

This sample code installs version 3.0.0-1 of the R package on Mac.

```r
PKG_URL <- "https://github.com/microsoft/LightGBM/releases/download/v3.0.0rc1/lightgbm-3.0.0-1-r40-macos.tgz"

local_file <- paste0("lightgbm.", tools::file_ext(PKG_URL))

download.file(
    url = PKG_URL
    , destfile = local_file
)
install.packages(
    pkgs = local_file
    , type = "binary"
    , repos = NULL
)
```

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

Preparing a CRAN Package
------------------------

This section is primarily for maintainers, but may help users and contributors to understand the structure of the R package.

Most of `LightGBM` uses `CMake` to handle tasks like setting compiler and linker flags, including header file locations, and linking to other libraries. Because CRAN packages typically do not assume the presence of `CMake`, the R package uses an alternative method that is in the CRAN-supported toolchain for building R packages with C++ code: `Autoconf`.

For more information on this approach, see ["Writing R Extensions"](https://cran.r-project.org/doc/manuals/r-release/R-exts.html#Configure-and-cleanup).

### Build a CRAN Package

From the root of the repository, run the following.

```shell
sh build-cran-package.sh
```

This will create a file `lightgbm_${VERSION}.tar.gz`, where `VERSION` is the version of `LightGBM`.

Alternatively, GitHub Actions can generate this file for you. On a pull request, go to the "Files changed" tab and create a comment with this phrase:

> /gha build r-artifacts

Go to https://github.com/microsoft/LightGBM/actions, and find the most recent run of the "R artifact builds" workflow. If it ran successfully, you'll find a download link for the package (in `.zip` format) in that run's "Artifacts" section.

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

1. Edit `configure.ac`
2. Create `configure` with `autoconf`. Do not edit it by hand. This file must be generated on Ubuntu 18.04.

    If you have an Ubuntu 18.04 environment available, run the provided script from the root of the `LightGBM` repository.

    ```shell
    ./R-package/recreate-configure.sh
    ```

    If you do not have easy access to an Ubuntu 18.04 environment, the `configure` script can be generated using Docker by running the code below from the root of this repo.

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

### Build Precompiled Binaries of the CRAN Package

This section is mainly for maintainers. As long as the R package is not available on CRAN (which will build precompiled binaries automatically) you may want to build precompiled versions of the R package manually, since these will be easier for users to install.

For more details, see ["Writing R Extensions"](https://cran.r-project.org/doc/manuals/r-release/R-exts.html#Building-binary-packages).

Packages built like this will only work for the minor version of R used to build them. They may or may not work across different versions of operating systems.

**Mac**

Binary produced: `lightgbm-${VERSION}-r40-macos.tgz`.

```shell
LGB_VERSION="3.0.0-1"
sh build-cran-package.sh
R CMD INSTALL --build lightgbm_${LGB_VERSION}.tar.gz
mv \
    lightgbm_${LGB_VERSION}.tgz \
    lightgbm-${LGB_VERSION}-r40-macos.tgz
```

**Linux**

Binary produced: `lightgbm-${VERSION}-r40-linux.tgz`.

You can access a Linux system that has R and its build toolchain installed with the `rocker` Docker images.

```shell
R_VERSION=4.0.2

docker run \
    -v $(pwd):/opt/LightGBM \
    -it rocker/verse:${R_VERSION} \
        /bin/bash
```

From inside that container, the commands to create a precompiled binary are very similar.

```shell
cd /opt/LightGBM
LGB_VERSION="3.0.0-1"
sh build-cran-package.sh
R CMD INSTALL --build lightgbm_${LGB_VERSION}.tar.gz
mv \
    lightgbm_${LGB_VERSION}_R_*-linux-gnu.tar.gz \
    lightgbm-${LGB_VERSION}-r40-linux.tgz
```

Exit the container, and the binary package should still be there on the host system.

```shell
exit
```

**Windows**

Binary produced: `lightgbm-${VERSION}-r40-windows.zip`.

```shell
LGB_VERSION="3.0.0-1"
sh build-cran-package.sh
R CMD INSTALL --build lightgbm_${LGB_VERSION}.tar.gz
mv \
    lightgbm_${LGB_VERSION}.tgz \
    lightgbm-${LGB_VERSION}-r40-windows.zip
```

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

#### UBSAN

All packages uploaded to CRAN must pass a build using `gcc` instrumented with two sanitizers: the Address Sanitizer (ASAN) and the Undefined Behavior Sanitizer (UBSAN). For more background, see [this blog post](http://dirk.eddelbuettel.com/code/sanitizers.html).

You can replicate these checks locally using Docker.

```shell
docker run \
    -v $(pwd):/opt/LightGBM \
    -it rhub/rocker-gcc-san \
    /bin/bash

cd /opt/LightGBM
Rscript -e "install.packages(c('R6', 'data.table', 'jsonlite', 'testthat'), repos = 'https://cran.rstudio.com')"

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
    -it \
        wch1/r-debug

cd /opt/LightGBM
RDscriptvalgrind -e "install.packages(c('R6', 'data.table', 'jsonlite', 'testthat'), repos = 'https://cran.rstudio.com')"

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

These tests can also be triggered on any pull request by leaving a review on the "Files changed" tab in a pull request:

> /gha run r-valgrind

External (Unofficial) Repositories
----------------------------------

Projects listed here are not maintained or endorsed by the `LightGBM` development team, but may offer some features currently missing from the main R package.

* [lightgbm.py](https://github.com/kapsner/lightgbm.py): This R package offers a wrapper built with `reticulate`, a package used to call Python code from R. If you are comfortable with the added installation complexity of installing `lightgbm`'s Python package and the performance cost of passing data between R and Python, you might find that this package offers some features that are not yet available in the native `lightgbm` R package.

Known Issues
------------

For information about known issues with the R package, see the [R-package section of LightGBM's main FAQ page](https://lightgbm.readthedocs.io/en/latest/FAQ.html#r-package).
