Quick Start
===========

This is a quick start guide for LightGBM CLI version.

Follow the `Installation Guide <./Installation-Guide.rst>`__ to install LightGBM first.

**List of other helpful links**

-  `Parameters <./Parameters.rst>`__

-  `Parameters Tuning <./Parameters-Tuning.rst>`__

-  `Python-package Quick Start <./Python-Intro.rst>`__

-  `Python API <./Python-API.rst>`__

Training Data Format
--------------------

LightGBM supports input data file with `CSV`_, `TSV`_ and `LibSVM`_ formats.

Label is the data of first column, and there is no header in the file.

Categorical Feature Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~

update 12/5/2016:

LightGBM can use categorical feature directly (without one-hot coding).
The experiment on `Expo data`_ shows about 8x speed-up compared with one-hot coding.

For the setting details, please refer to `Parameters <./Parameters.rst>`__.

Weight and Query/Group Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~

LightGBM also support weighted training, it needs an additional `weight data <./Parameters.rst#io-parameters>`__.
And it needs an additional `query data <./Parameters.rst#io-parameters>`_ for ranking task.

update 11/3/2016:

1. support input with header now

2. can specific label column, weight column and query/group id column.
   Both index and column are supported

3. can specific a list of ignored columns

Parameter Quick Look
--------------------

The parameter format is ``key1=value1 key2=value2 ...``.
And parameters can be in both config file and command line.

Some important parameters:

- ``config``, default=\ ``""``, type=string, alias=\ ``config_file``

  - path to config file

-  ``task``, default=\ ``train``, type=enum, options=\ ``train``, ``predict``, ``convert_model``

   -  ``train``, alias=\ ``training``, for training

   -  ``predict``, alias=\ ``prediction``, ``test``, for prediction.

   -  ``convert_model``, for converting model file into if-else format, see more information in `Convert model parameters <./Parameters.rst#convert-model-parameters>`__

-  ``application``, default=\ ``regression``, type=enum,
   options=\ ``regression``, ``regression_l1``, ``huber``, ``fair``, ``poisson``, ``quantile``, ``quantile_l2``,
   ``binary``, ``multiclass``, ``multiclassova``, ``xentropy``, ``xentlambda``, ``lambdarank``,
   alias=\ ``objective``, ``app``

   -  regression application

      -  ``regression_l2``, L2 loss, alias=\ ``regression``, ``mean_squared_error``, ``mse``

      -  ``regression_l1``, L1 loss, alias=\ ``mean_absolute_error``, ``mae``

      -  ``huber``, `Huber loss`_

      -  ``fair``, `Fair loss`_

      -  ``poisson``, `Poisson regression`_

      -  ``quantile``, `Quantile regression`_

      -  ``quantile_l2``, like the ``quantile``, but L2 loss is used instead

   -  ``binary``, binary `log loss`_ classification application

   -  multi-class classification application

      -  ``multiclass``, `softmax`_ objective function, ``num_class`` should be set as well

      -  ``multiclassova``, `One-vs-All`_ binary objective function, ``num_class`` should be set as well

   -  cross-entropy application

      -  ``xentropy``, objective function for cross-entropy (with optional linear weights), alias=\ ``cross_entropy``

      -  ``xentlambda``, alternative parameterization of cross-entropy, alias=\ ``cross_entropy_lambda``

      -  the label is anything in interval [0, 1]

   -  ``lambdarank``, `lambdarank`_ application

      -  the label should be ``int`` type in lambdarank tasks, and larger number represent the higher relevance (e.g. 0:bad, 1:fair, 2:good, 3:perfect)

      -  ``label_gain`` can be used to set the gain(weight) of ``int`` label

- ``boosting``, default=\ ``gbdt``, type=enum,
  options=\ ``gbdt``, ``rf``, ``dart``, ``goss``,
  alias=\ ``boost``, ``boosting_type``

  - ``gbdt``, traditional Gradient Boosting Decision Tree

  - ``rf``, Random Forest

  - ``dart``, `Dropouts meet Multiple Additive Regression Trees`_

  - ``goss``, Gradient-based One-Side Sampling

- ``data``, default=\ ``""``, type=string, alias=\ ``train``, ``train_data``

  - training data, LightGBM will train from this data

- ``valid``, default=\ ``""``, type=multi-string, alias=\ ``test``, ``valid_data``, ``test_data``

  - validation/test data, LightGBM will output metrics for these data

  - support multi validation data, separate by ``,``

- ``num_iterations``, default=\ ``100``, type=int,
  alias=\ ``num_iteration``, ``num_tree``, ``num_trees``, ``num_round``, ``num_rounds``, ``num_boost_round``

  - number of boosting iterations/trees

- ``learning_rate``, default=\ ``0.1``, type=double, alias=\ ``shrinkage_rate``

  - shrinkage rate

- ``num_leaves``, default=\ ``31``, type=int, alias=\ ``num_leaf``

  - number of leaves in one tree

-  ``tree_learner``, default=\ ``serial``, type=enum, options=\ ``serial``, ``feature``, ``data``, ``voting``, alias=\ ``tree``

   -  ``serial``, single machine tree learner

   -  ``feature``, alias=\ ``feature_parallel``, feature parallel tree learner

   -  ``data``, alias=\ ``data_parallel``, data parallel tree learner

   -  ``voting``, alias=\ ``voting_parallel``, voting parallel tree learner

   -  refer to `Parallel Learning Guide <./Parallel-Learning-Guide.rst>`__ to get more details

- ``num_threads``, default=\ ``OpenMP_default``, type=int, alias=\ ``num_thread``, ``nthread``

  - number of threads for LightGBM

  - for the best speed, set this to the number of **real CPU cores**,
    not the number of threads (most CPU using `hyper-threading`_ to generate 2 threads per CPU core)

  - for parallel learning, should not use full CPU cores since this will cause poor performance for the network

- ``max_depth``, default=\ ``-1``, type=int

  - limit the max depth for tree model.
    This is used to deal with overfit when ``#data`` is small.
    Tree still grow by leaf-wise

  - ``< 0`` means no limit

- ``min_data_in_leaf``, default=\ ``20``, type=int, alias=\ ``min_data_per_leaf`` , ``min_data``, ``min_child_samples``

  - minimal number of data in one leaf. Can use this to deal with over-fitting

- ``min_sum_hessian_in_leaf``, default=\ ``1e-3``, type=double,
  alias=\ ``min_sum_hessian_per_leaf``, ``min_sum_hessian``, ``min_hessian``, ``min_child_weight``

  - minimal sum hessian in one leaf. Like ``min_data_in_leaf``, it can be used to deal with over-fitting

For all parameters, please refer to `Parameters <./Parameters.rst>`__.

Run LightGBM
------------

For Windows:

::

    lightgbm.exe config=your_config_file other_args ...

For Unix:

::

    ./lightgbm config=your_config_file other_args ...

Parameters can be both in the config file and command line, and the parameters in command line have higher priority than in config file.
For example, following command line will keep ``num_trees=10`` and ignore the same parameter in config file.

::

    ./lightgbm config=train.conf num_trees=10

Examples
--------

-  `Binary Classification <https://github.com/Microsoft/LightGBM/tree/master/examples/binary_classification>`__

-  `Regression <https://github.com/Microsoft/LightGBM/tree/master/examples/regression>`__

-  `Lambdarank <https://github.com/Microsoft/LightGBM/tree/master/examples/lambdarank>`__

-  `Parallel Learning <https://github.com/Microsoft/LightGBM/tree/master/examples/parallel_learning>`__

.. _CSV: https://en.wikipedia.org/wiki/Comma-separated_values

.. _TSV: https://en.wikipedia.org/wiki/Tab-separated_values

.. _LibSVM: https://www.csie.ntu.edu.tw/~cjlin/libsvm/

.. _Expo data: http://stat-computing.org/dataexpo/2009/

.. _Huber loss: https://en.wikipedia.org/wiki/Huber_loss

.. _Fair loss: https://www.kaggle.com/c/allstate-claims-severity/discussion/24520

.. _Poisson regression: https://en.wikipedia.org/wiki/Poisson_regression

.. _Quantile regression: https://en.wikipedia.org/wiki/Quantile_regression

.. _log loss: https://www.kaggle.com/wiki/LogLoss

.. _softmax: https://en.wikipedia.org/wiki/Softmax_function

.. _One-vs-All: https://en.wikipedia.org/wiki/Multiclass_classification#One-vs.-rest

.. _lambdarank: https://papers.nips.cc/paper/2971-learning-to-rank-with-nonsmooth-cost-functions.pdf

.. _Dropouts meet Multiple Additive Regression Trees: https://arxiv.org/abs/1505.01866

.. _hyper-threading: https://en.wikipedia.org/wiki/Hyper-threading
