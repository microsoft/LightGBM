LightGBM R Package
==================

Installation
------------

Windows users may need to run with administrator rights (either R or the command prompt, depending on the way you are installing this package). Rtools must be installed.

You can use a command prompt to install via command line:

```
cd R-package
R CMD INSTALL --build  .
```

You can also install directly from R using the repository with `devtools`:

```r
devtools::install_github("Microsoft/LightGBM", subdir = "R-package")
```

To install LightGBM from a specific commit, you can specify the reference, such as the following to install the first release of the R package for LightGBM:

```r
devtools::install_github("Microsoft/LightGBM", ref = "1b7643b", subdir = "R-package")
```

You can check quickly if your LightGBM R package is working by running the following:

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

* Please visit [demo](demo).
