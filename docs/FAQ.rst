LightGBM FAQ
============

Contents
~~~~~~~~

-  `Critical <#critical>`__

-  `LightGBM <#lightgbm>`__

-  `R-package <#r-package>`__

-  `Python-package <#python-package>`__

--------------

Critical
~~~~~~~~

You encountered a critical issue when using LightGBM (crash, prediction error, non sense outputs...). Who should you contact?

If your issue is not critical, just post an issue in `Microsoft/LightGBM repository <https://github.com/Microsoft/LightGBM/issues>`__.

If it is a critical issue, identify first what error you have:

-  Do you think it is reproducible on CLI (command line interface), R, and/or Python?

-  Is it specific to a wrapper? (R or Python?)

-  Is it specific to the compiler? (gcc versions? MinGW versions?)

-  Is it specific to your Operating System? (Windows? Linux?)

-  Are you able to reproduce this issue with a simple case?

-  Are you able to (not) reproduce this issue after removing all optimization flags and compiling LightGBM in debug mode?

Depending on the answers, while opening your issue, feel free to ping (just mention them with the arobase (@) symbol) appropriately so we can attempt to solve your problem faster:

-  `@guolinke <https://github.com/guolinke>`__ (C++ code / R-package / Python-package)
-  `@chivee <https://github.com/chivee>`__ (C++ code / Python-package)
-  `@Laurae2 <https://github.com/Laurae2>`__ (R-package)
-  `@wxchan <https://github.com/wxchan>`__ (Python-package)
-  `@henry0312 <https://github.com/henry0312>`__ (Python-package)
-  `@StrikerRUS <https://github.com/StrikerRUS>`__ (Python-package)
-  `@huanzhang12 <https://github.com/huanzhang12>`__ (GPU support)

Remember this is a free/open community support. We may not be available 24/7 to provide support.

--------------

LightGBM
~~~~~~~~

-  **Question 1**: Where do I find more details about LightGBM parameters?

-  **Solution 1**: Take a look at `Parameters <./Parameters.rst>`__ and `Laurae++/Parameters <https://sites.google.com/view/lauraepp/parameters>`__ website.

--------------

-  **Question 2**: On datasets with million of features, training do not start (or starts after a very long time).

-  **Solution 2**: Use a smaller value for ``bin_construct_sample_cnt`` and a larger value for ``min_data``.

--------------

-  **Question 3**: When running LightGBM on a large dataset, my computer runs out of RAM.

-  **Solution 3**: Multiple solutions: set ``histogram_pool_size`` parameter to the MB you want to use for LightGBM (histogram\_pool\_size + dataset size = approximately RAM used),
   lower ``num_leaves`` or lower ``max_bin`` (see `Microsoft/LightGBM#562 <https://github.com/Microsoft/LightGBM/issues/562>`__).

--------------

-  **Question 4**: I am using Windows. Should I use Visual Studio or MinGW for compiling LightGBM?

-  **Solution 4**: It is recommended to `use Visual Studio <https://github.com/Microsoft/LightGBM/issues/542>`__ as its performance is higher for LightGBM.

--------------

-  **Question 5**: When using LightGBM GPU, I cannot reproduce results over several runs.

-  **Solution 5**: It is a normal issue, there is nothing we/you can do about,
   you may try to use ``gpu_use_dp = true`` for reproducibility (see `Microsoft/LightGBM#560 <https://github.com/Microsoft/LightGBM/pull/560#issuecomment-304561654>`__).
   You may also use CPU version.

--------------

-  **Question 6**: Bagging is not reproducible when changing the number of threads.

-  **Solution 6**: As LightGBM bagging is running multithreaded, its output is dependent on the number of threads used.
   There is `no workaround currently <https://github.com/Microsoft/LightGBM/issues/632>`__.

--------------

-  **Question 7**: I tried to use Random Forest mode, and LightGBM crashes!

-  **Solution 7**: It is by design.
   You must use ``bagging_fraction`` and ``feature_fraction`` different from 1, along with a ``bagging_freq``.
   See `this thread <https://github.com/Microsoft/LightGBM/issues/691>`__ as an example.

--------------

-  **Question 8**: CPU are not kept busy (like 10% CPU usage only) in Windows when using LightGBM on very large datasets with many core systems.

-  **Solution 8**: Please use `Visual Studio <https://www.visualstudio.com/downloads/>`__
   as it may be `10x faster than MinGW <https://github.com/Microsoft/LightGBM/issues/749>`__ especially for very large trees.

--------------

R-package
~~~~~~~~~

-  **Question 1**: Any training command using LightGBM does not work after an error occurred during the training of a previous LightGBM model.

-  **Solution 1**: Run ``lgb.unloader(wipe = TRUE)`` in the R console, and recreate the LightGBM datasets (this will wipe all LightGBM-related variables).
   Due to the pointers, choosing to not wipe variables will not fix the error.
   This is a known issue: `Microsoft/LightGBM#698 <https://github.com/Microsoft/LightGBM/issues/698>`__.

--------------

-  **Question 2**: I used ``setinfo``, tried to print my ``lgb.Dataset``, and now the R console froze!

-  **Solution 2**: Avoid printing the ``lgb.Dataset`` after using ``setinfo``.
   This is a known bug: `Microsoft/LightGBM#539 <https://github.com/Microsoft/LightGBM/issues/539>`__.

--------------

Python-package
~~~~~~~~~~~~~~

-  **Question 1**: I see error messages like this when install from GitHub using ``python setup.py install``.

   ::

       error: Error: setup script specifies an absolute path:
       /Users/Microsoft/LightGBM/python-package/lightgbm/../../lib_lightgbm.so
       setup() arguments must *always* be /-separated paths relative to the setup.py directory, *never* absolute paths.

-  **Solution 1**: This error should be solved in latest version.
   If you still meet this error, try to remove ``lightgbm.egg-info`` folder in your Python-package and reinstall,
   or check `this thread on stackoverflow <http://stackoverflow.com/questions/18085571/pip-install-error-setup-script-specifies-an-absolute-path>`__.

--------------

-  **Question 2**: I see error messages like

   ::

       Cannot get/set label/weight/init_score/group/num_data/num_feature before construct dataset

   but I've already constructed dataset by some code like

   ::

       train = lightgbm.Dataset(X_train, y_train)

   or error messages like

   ::

       Cannot set predictor/reference/categorical feature after freed raw data, set free_raw_data=False when construct Dataset to avoid this.

-  **Solution 2**: Because LightGBM constructs bin mappers to build trees, and train and valid Datasets within one Booster share the same bin mappers,
   categorical features and feature names etc., the Dataset objects are constructed when construct a Booster.
   And if you set ``free_raw_data=True`` (default), the raw data (with Python data struct) will be freed.
   So, if you want to:

   -  get label(or weight/init\_score/group) before construct dataset, it's same as get ``self.label``

   -  set label(or weight/init\_score/group) before construct dataset, it's same as ``self.label=some_label_array``

   -  get num\_data(or num\_feature) before construct dataset, you can get data with ``self.data``,
      then if your data is ``numpy.ndarray``, use some code like ``self.data.shape``

   -  set predictor(or reference/categorical feature) after construct dataset,
      you should set ``free_raw_data=False`` or init a Dataset object with the same raw data
