Python Package Introduction
===========================

This document gives a basic walkthrough of LightGBM Python-package.

**List of other helpful links**

-  `Python Examples <https://github.com/Microsoft/LightGBM/tree/master/examples/python-guide>`__

-  `Python API <./Python-API.rst>`__

-  `Parameters Tuning <./Parameters-Tuning.rst>`__

Install
-------

Install Python-package dependencies,
``setuptools``, ``wheel``, ``numpy`` and ``scipy`` are required, ``scikit-learn`` is required for sklearn interface and recommended:

::

    pip install setuptools wheel numpy scipy scikit-learn -U

Refer to `Python-package`_ folder for the installation guide.

To verify your installation, try to ``import lightgbm`` in Python:

::

    import lightgbm as lgb

Data Interface
--------------

The LightGBM Python module is able to load data from:

-  libsvm/tsv/csv/txt format file

-  Numpy 2D array, pandas object

-  LightGBM binary file

The data is stored in a ``Dataset`` object.

**To load a libsvm text file or a LightGBM binary file into Dataset:**

.. code:: python

    train_data = lgb.Dataset('train.svm.bin')

**To load a numpy array into Dataset:**

.. code:: python

    data = np.random.rand(500, 10)  # 500 entities, each contains 10 features
    label = np.random.randint(2, size=500)  # binary target
    train_data = lgb.Dataset(data, label=label)

**To load a scpiy.sparse.csr\_matrix array into Dataset:**

.. code:: python

    csr = scipy.sparse.csr_matrix((dat, (row, col)))
    train_data = lgb.Dataset(csr)

**Saving Dataset into a LightGBM binary file will make loading faster:**

.. code:: python

    train_data = lgb.Dataset('train.svm.txt')
    train_data.save_binary('train.bin')

**Create validation data:**

.. code:: python

    test_data = train_data.create_valid('test.svm')

or

.. code:: python

    test_data = lgb.Dataset('test.svm', reference=train_data)

In LightGBM, the validation data should be aligned with training data.

**Specific feature names and categorical features:**

.. code:: python

    train_data = lgb.Dataset(data, label=label, feature_name=['c1', 'c2', 'c3'], categorical_feature=['c3'])

LightGBM can use categorical features as input directly.
It doesn't need to convert to one-hot coding, and is much faster than one-hot coding (about 8x speed-up).

**Note**: You should convert your categorical features to ``int`` type before you construct ``Dataset``.

**Weights can be set when needed:**

.. code:: python

    w = np.random.rand(500, )
    train_data = lgb.Dataset(data, label=label, weight=w)

or

.. code:: python

    train_data = lgb.Dataset(data, label=label)
    w = np.random.rand(500, )
    train_data.set_weight(w)

And you can use ``Dataset.set_init_score()`` to set initial score, and ``Dataset.set_group()`` to set group/query data for ranking tasks.

**Memory efficent usage:**

The ``Dataset`` object in LightGBM is very memory-efficient, due to it only need to save discrete bins.
However, Numpy/Array/Pandas object is memory cost.
If you concern about your memory consumption, you can save memory according to following:

1. Let ``free_raw_data=True`` (default is ``True``) when constructing the ``Dataset``

2. Explicit set ``raw_data=None`` after the ``Dataset`` has been constructed

3. Call ``gc``

Setting Parameters
------------------

LightGBM can use either a list of pairs or a dictionary to set `Parameters <./Parameters.rst>`__.
For instance:

-  Booster parameters:

   .. code:: python

       param = {'num_leaves':31, 'num_trees':100, 'objective':'binary'}
       param['metric'] = 'auc'

-  You can also specify multiple eval metrics:

   .. code:: python

       param['metric'] = ['auc', 'binary_logloss']

Training
--------

Training a model requires a parameter list and data set:

.. code:: python

    num_round = 10
    bst = lgb.train(param, train_data, num_round, valid_sets=[test_data])

After training, the model can be saved:

.. code:: python

    bst.save_model('model.txt')

The trained model can also be dumped to JSON format:

.. code:: python

    json_model = bst.dump_model()

A saved model can be loaded:

.. code:: python

    bst = lgb.Booster(model_file='model.txt')  #init model

CV
--

Training with 5-fold CV:

.. code:: python

    num_round = 10
    lgb.cv(param, train_data, num_round, nfold=5)

Early Stopping
--------------

If you have a validation set, you can use early stopping to find the optimal number of boosting rounds.
Early stopping requires at least one set in ``valid_sets``. If there is more than one, it will use all of them:

.. code:: python

    bst = lgb.train(param, train_data, num_round, valid_sets=valid_sets, early_stopping_rounds=10)
    bst.save_model('model.txt', num_iteration=bst.best_iteration)

The model will train until the validation score stops improving.
Validation error needs to improve at least every ``early_stopping_rounds`` to continue training.

If early stopping occurs, the model will have an additional field: ``bst.best_iteration``.
Note that ``train()`` will return a model from the best iteration.

This works with both metrics to minimize (L2, log loss, etc.) and to maximize (NDCG, AUC).
Note that if you specify more than one evaluation metric, all of them will be used for early stopping.

Prediction
----------

A model that has been trained or loaded can perform predictions on data sets:

.. code:: python

    # 7 entities, each contains 10 features
    data = np.random.rand(7, 10)
    ypred = bst.predict(data)

If early stopping is enabled during training, you can get predictions from the best iteration with ``bst.best_iteration``:

.. code:: python

    ypred = bst.predict(data, num_iteration=bst.best_iteration)

.. _Python-package: https://github.com/Microsoft/LightGBM/tree/master/python-package
