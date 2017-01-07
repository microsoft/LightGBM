LightGBM FAQ
=======================

###Catalog

- [Python-package](FAQ.md#python-package)

###Python-package

- **Question 1**: I see error messages like this when install from github using `python setup.py install`.

    ```
    error: Error: setup script specifies an absolute path:

    /Users/Microsoft/LightGBM/python-package/lightgbm/../../lib_lightgbm.so

    setup() arguments must *always* be /-separated paths relative to the
    setup.py directory, *never* absolute paths.
    ```

- **Solution 1**: please check [this thread on stackoverflow](http://stackoverflow.com/questions/18085571/pip-install-error-setup-script-specifies-an-absolute-path).

- **Question 2**: I see error messages like `Cannot get/set label/weight/init_score/group/num_data/num_feature before construct dataset`, but I already contruct dataset by some code like `train = lightgbm.Dataset(X_train, y_train)`, or error messages like `Cannot set predictor/reference/categorical feature after freed raw data, set free_raw_data=False when construct Dataset to avoid this.`.

- **Solution 2**: Because LightGBM contructs bin mappers to build trees, and train and valid Datasets within one Booster share the same bin mappers, categorical features and feature names etc., the Dataset objects are constructed when contruct a Booster. And if you set free_raw_data=True (default), the raw data (with python data struct) will be freed. So, if you want to:

  + get label(or weight/init_score/group) before contruct dataset, it's same as get `self.label`
  + set label(or weight/init_score/group) before contruct dataset, it's same as `self.label=some_label_array`
  + get num_data(or num_feature) before contruct dataset, you can get data with `self.data`, then if your data is `numpy.ndarray`, use some code like `self.data.shape`
  + set predictor(or reference/categorical feature) after contruct dataset, you should set free_raw_data=False or init a Dataset object with the same raw data
