LightGBM R Package
==================

Installation
------------

Command line installation:

```
cd R-package
R CMD INSTALL --build  .
```

Examples
------------

* Please visit [demo](demo).

Windows installation example
------------

Here is a full guide for installing LightGBM for R in Windows, using [Visual Studio 2015](https://www.visualstudio.com/downloads/) and [Git Bash](https://git-for-windows.github.io/). We suppose you will install LightGBM in `C:/LightGBM`.

This applies to Windows only. Linux users can just compile "out of the box" LightGBM with the gcc tool chain

LightGBM uses Visual Studio (2013 or higher) to build in Windows. If you do not have Visual Studio, follow this: download Visual Studio 2015 Community. It is free. When installing Visual Studio Community, install it with the Visual C++ additions (custom install, select the first box which has 3 subboxes - it should say you will install the Windows SDK - ignore the update failure error at the end). Prepare at least 8GB of free drive space.

Once you are done installing Visual Studio 2015 Community, reboot your computer.

Now, clone the LightGBM repository by doing in Git Bash:

```
cd C:/
git clone --recursive https://github.com/Microsoft/LightGBM
```

Now the steps in Visual Studio 2015:

* Under C:/xgboost/LightGBM/windows, double click LightGBM.sln to open it in Visual Studio.
* Accept any warning pop up about project versioning issues (Upgrade VC++ Compiler and Libraries --> OK).
* Wait one minute for the loading.
* On the Solution Explorer, click "Solution 'LightGBM' (1 project)"
* On the bottom right tab (Properties), change the "Active config" to "Release|x64" (default is "Debug_mpi|x64")
* Compile the solution by pressing Ctrl+Shift+B (or click Build > Build Solution).
* Should everything be correct, you now have LightGBM compiled under C:\LightGBM\windows\x64\Release
* If you get an error while building (Windows SDK version), then you will need the correct SDK for your OS. Start Visual Studio from scratch, click "New Project", select "Visual C++" and click "Install Visual C++ 2015 Tools for Windows Desktop". Then, attempt to build LightGBM.

To avoid using the command prompt, start R (preferably as an administrator) and run this command:

```r
setwd("C:/")
devtools::install("R-package")
```

If you did not have a compilation error, then the installation is successful and you can try a demo straight after by running the following:

```r
library(lightgbm)
data(agaricus.train, package='lightgbm')
train <- agaricus.train
dtrain <- lgb.Dataset(train$data, label=train$label)
params <- list(objective="regression", metric="l2")
model <- lgb.cv(params, dtrain, 10, nfold=5, min_data=1, learning_rate=1, early_stopping_rounds=10)
```

If this does not work, you need to install these two additional packages:

```r
install.packages(c("R6", "Matrix"))
```
