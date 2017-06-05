LightGBM R Package
==================

Installation
------------

Windows users may need to run with administrator rights (either R or the command prompt, depending on the way you are installing this package). Rtools must be installed for Windows. Linux users might require the appropriate user write permissions for packages.

You need to install *cmake* first. The default compiler in Windows is *Visual Studio*, you can use *MinGW* by set ```use_mingw <- TRUE``` in ```R-package/src/install.libs.R``` (We recommend *Visual Studio* for its better multi-threading efficency in Windows).

Install by following command:

```
git clone --recursive https://github.com/Microsoft/LightGBM
cd LightGBM/R-package
Rscript build_package.R
R CMD INSTALL lightgbm_0.1.tar.gz
``` 

Set ```use_gpu <- TRUE``` in ```R-package/src/install.libs.R``` can enable the build with GPU support (Need to install *Boost* and *OpenCL* first, details can be found in [gpu-support](https://github.com/Microsoft/LightGBM/wiki/Installation-Guide#with-gpu-support)).

You can also install directly from R using the repository with `devtools`:

```r
devtools::install_github("Microsoft/LightGBM", subdir = "R-package")
```


When your package installation is done, you can check quickly if your LightGBM R package is working by running the following:

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

Please visit [demo](demo):

* [Basic walkthrough of wrappers](demo/basic_walkthrough.R)
* [Boosting from existing prediction](demo/boost_from_prediction.R)
* [Early Stopping](demo/early_stopping.R)
* [Cross Validation](demo/cross_validation.R)
* [Multiclass Training/Prediction](demo/multiclass.R)
* [Leaf (in)Stability](demo/leaf_stability.R)
* [Weight-Parameter Adjustment Relationship](demo/weight_param.R)
