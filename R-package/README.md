LightGBM R Package
==================

[![Build Status](https://travis-ci.org/Microsoft/LightGBM.svg?branch=master)](https://travis-ci.org/Microsoft/LightGBM)
[![Windows Build Status](https://ci.appveyor.com/api/projects/status/1ys5ot401m0fep6l/branch/master?svg=true)](https://ci.appveyor.com/project/guolinke/lightgbm/branch/master)
[![Documentation Status](https://readthedocs.org/projects/lightgbm/badge/?version=latest)](https://lightgbm.readthedocs.io/)
[![GitHub Issues](https://img.shields.io/github/issues/Microsoft/LightGBM.svg)](https://github.com/Microsoft/LightGBM/issues)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Microsoft/LightGBM/blob/master/LICENSE)
[![Python Versions](https://img.shields.io/pypi/pyversions/lightgbm.svg)](https://pypi.python.org/pypi/lightgbm)
[![PyPI Version](https://badge.fury.io/py/lightgbm.svg)](https://badge.fury.io/py/lightgbm)
[![Join the chat at https://gitter.im/Microsoft/LightGBM](https://badges.gitter.im/Microsoft/LightGBM.svg)](https://gitter.im/Microsoft/LightGBM?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Slack](https://lightgbm-slack-autojoin.herokuapp.com/badge.svg)](https://lightgbm-slack-autojoin.herokuapp.com/)

LightGBM, Light Gradient Boosting Machine
=========================================

LightGBM is a gradient boosting framework that uses tree based learning algorithms. It is designed to be distributed and efficient with the following advantages:

- Faster training speed and higher efficiency
- Lower memory usage
- Better accuracy
- Parallel and GPU learning supported
- Capable of handling large-scale data

For more details, please refer to [Features](https://github.com/Microsoft/LightGBM/blob/master/docs/Features.rst).

[Comparison experiments](https://github.com/Microsoft/LightGBM/blob/master/docs/Experiments.rst#comparison-experiment) on public datasets show that LightGBM can outperform existing boosting frameworks on both efficiency and accuracy, with significantly lower memory consumption. What's more, the [parallel experiments](https://github.com/Microsoft/LightGBM/blob/master/docs/Experiments.rst#parallel-experiment) show that LightGBM can achieve a linear speed-up by using multiple machines for training in specific settings.

News
----

08/15/2017 : Optimal split for categorical features.

07/13/2017 : [Gitter](https://gitter.im/Microsoft/LightGBM) is available.

06/20/2017 : Python-package is on [PyPI](https://pypi.python.org/pypi/lightgbm) now.

06/09/2017 : [LightGBM Slack team](https://lightgbm.slack.com) is available.

05/03/2017 : LightGBM v2 stable release.

04/10/2017 : LightGBM supports GPU-accelerated tree learning now. Please read our [GPU Tutorial](./docs/GPU-Tutorial.rst) and [Performance Comparison](./docs/GPU-Performance.rst).

02/20/2017 : Update to LightGBM v2.

02/12/2017: LightGBM v1 stable release.

01/08/2017 : Release [**R-package**](https://github.com/Microsoft/LightGBM/tree/master/R-package) beta version, welcome to have a try and provide feedback.

12/05/2016 : **Categorical Features as input directly** (without one-hot coding). 

12/02/2016 : Release [**Python-package**](https://github.com/Microsoft/LightGBM/tree/master/python-package) beta version, welcome to have a try and provide feedback.

More detailed update logs : [Key Events](https://github.com/Microsoft/LightGBM/blob/master/docs/Key-Events.md).

External (unofficial) Repositories
----------------------------------

Julia Package: https://github.com/Allardvm/LightGBM.jl

JPMML: https://github.com/jpmml/jpmml-lightgbm

Get Started and Documentation
-----------------------------

Install by following the [guide](https://github.com/Microsoft/LightGBM/blob/master/docs/Installation-Guide.rst) for the command line program, [Python-package](https://github.com/Microsoft/LightGBM/tree/master/python-package) or [R-package](https://github.com/Microsoft/LightGBM/tree/master/R-package). Then please see the [Quick Start](https://github.com/Microsoft/LightGBM/blob/master/docs/Quick-Start.rst) guide.

Our primary documentation is at https://lightgbm.readthedocs.io/ and is generated from this repository.

Next you may want to read:

* [**Examples**](https://github.com/Microsoft/LightGBM/tree/master/examples) showing command line usage of common tasks
* [**Features**](https://github.com/Microsoft/LightGBM/blob/master/docs/Features.rst) and algorithms supported by LightGBM
* [**Parameters**](https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst) is an exhaustive list of customization you can make
* [**Parallel Learning**](https://github.com/Microsoft/LightGBM/blob/master/docs/Parallel-Learning-Guide.rst) and [**GPU Learning**](https://github.com/Microsoft/LightGBM/blob/master/docs/GPU-Tutorial.rst) can speed up computation
* [**Laurae++ interactive documentation**](https://sites.google.com/view/lauraepp/parameters) is a detailed guide for hyperparameters

Documentation for contributors:

* [**How we update readthedocs.io**](https://github.com/Microsoft/LightGBM/blob/master/docs/README.rst)
* Check out the [**Development Guide**](https://github.com/Microsoft/LightGBM/blob/master/docs/Development-Guide.rst).

Support
-------

* Ask a question [on Stack Overflow with the `lightgbm` tag](https://stackoverflow.com/questions/ask?tags=lightgbm), we monitor this for new questions.
* Discuss on the [LightGBM Gitter](https://gitter.im/Microsoft/LightGBM).
* Discuss on the [LightGBM Slack team](https://lightgbm.slack.com).
  * Use [this invite link](https://lightgbm-slack-autojoin.herokuapp.com/) to join the team.
* Open **bug reports** and **feature requests** (not questions) on [GitHub issues](https://github.com/Microsoft/LightGBM/issues).

How to Contribute
-----------------

LightGBM has been developed and used by many active community members. Your help is very valuable to make it better for everyone.

- Check out [call for contributions](https://github.com/Microsoft/LightGBM/issues?q=is%3Aissue+is%3Aopen+label%3Acall-for-contribution) to see what can be improved, or open an issue if you want something.
- Contribute to the [tests](https://github.com/Microsoft/LightGBM/tree/master/tests) to make it more reliable.
- Contribute to the [documents](https://github.com/Microsoft/LightGBM/tree/master/docs) to make it clearer for everyone.
- Contribute to the [examples](https://github.com/Microsoft/LightGBM/tree/master/examples) to share your experience with other users.
- Open issue if you met problems during development.

Microsoft Open Source Code of Conduct
-------------------------------------

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

Reference Paper
---------------

Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu. [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree). In Advances in Neural Information Processing Systems (NIPS), pp. 3149-3157. 2017.

Installation
------------

### Preparation

You need to install git and [CMake](https://cmake.org/) first.

Note: 32-bit R/Rtools is not supported.

#### Windows Preparation

Installing [Rtools](https://cran.r-project.org/bin/windows/Rtools/) is mandatory, and only support the 64-bit version. It requires to add to PATH the Rtools MinGW64 folder, if it was not done automatically during installation.

The default compiler is Visual Studio (or [MS Build](https://www.visualstudio.com/downloads/)) in Windows, with an automatic fallback to Rtools or any [MinGW64](https://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win64/Personal%20Builds/mingw-builds/) (x86_64-posix-seh) available (this means if you have only Rtools and CMake, it will compile fine).

To force the usage of Rtools / MinGW, you can set `use_mingw` to `TRUE` in `R-package/src/install.libs.R`.

For users who wants to install online with GPU or want to choose a specific compiler, please check the end of this document for installation using a helper package ([Laurae2/lgbdl](https://github.com/Laurae2/lgbdl/)).

**Warning for Windows users**: it is recommended to use *Visual Studio* for its better multi-threading efficency in Windows for many core systems. For very simple systems (dual core computers or worse), MinGW64 is recommended for maximum performance. If you do not know what to choose, it is recommended to use [Visual Studio](https://www.visualstudio.com/downloads/), the default compiler. **Do not try using MinGW in Windows on many core systems. It may result in 10x slower results than Visual Studio.**

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

Please visit [demo](https://github.com/Microsoft/LightGBM/tree/master/R-package/demo):

* [Basic walkthrough of wrappers](articles/basic_walkthrough.html)
* [Boosting from existing prediction](articles/boost_from_prediction.html)
* [Early Stopping](articles/early_stopping.html)
* [Cross Validation](articles/cross_validation.html)
* [Multiclass Training/Prediction](articles/multiclass.html)
* [Leaf (in)Stability](articles/leaf_stability.html)
* [Weight-Parameter Adjustment Relationship](articles/weight_param.html)
