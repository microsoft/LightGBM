LightGBM R Package
==================

Installation
------------

Windows users may need to run with administrator rights (either R or the command prompt, depending on the way you are installing this package). Rtools must be installed for Windows. Linux users might require the appropriate user write permissions for packages.

1. Following [Installation Guide](https://github.com/Microsoft/LightGBM/wiki/Installation-Guide) to build LightGBM first.
   For the windows user, please change the build config to ``DLL``.

2. Install by following command:

```
cd R-package
R CMD INSTALL --build  .
``` 

Note: build by ```devtools::install_github``` is not supportted any more.


If you want to build the self-contained R package:

1. Copy ```lib_lightgbm.so``` (or ```lib_lightgbm.dll``` in Windows) to ```R-package``` folder.

2. Build package via ```R CMD build --no-build-vignettes --binary .```

3. Install via ```R CMD INSTALL lightgbm_0.1.tar.gz```


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
