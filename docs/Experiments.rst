Experiments
===========

Comparison Experiment
---------------------

For the detailed experiment scripts and output logs, please refer to this `repo`_.

Data
^^^^

We used 5 datasets to conduct our comparison experiments. Details of data are listed in the following table:

+-----------+-----------------------+------------------------------------------------------------------------+-------------+----------+----------------------------------------------+
| Data      | Task                  | Link                                                                   | #Train\_Set | #Feature | Comments                                     |
+===========+=======================+========================================================================+=============+==========+==============================================+
| Higgs     | Binary classification | `link <https://archive.ics.uci.edu/ml/datasets/HIGGS>`__               | 10,500,000  | 28       | last 500,000 samples were used as test set   |
+-----------+-----------------------+------------------------------------------------------------------------+-------------+----------+----------------------------------------------+
| Yahoo LTR | Learning to rank      | `link <https://webscope.sandbox.yahoo.com/catalog.php?datatype=c>`__   | 473,134     | 700      | set1.train as train, set1.test as test       |
+-----------+-----------------------+------------------------------------------------------------------------+-------------+----------+----------------------------------------------+
| MS LTR    | Learning to rank      | `link <http://research.microsoft.com/en-us/projects/mslr/>`__          | 2,270,296   | 137      | {S1,S2,S3} as train set, {S5} as test set    |
+-----------+-----------------------+------------------------------------------------------------------------+-------------+----------+----------------------------------------------+
| Expo      | Binary classification | `link <http://stat-computing.org/dataexpo/2009/>`__                    | 11,000,000  | 700      | last 1,000,000 samples were used as test set |
+-----------+-----------------------+------------------------------------------------------------------------+-------------+----------+----------------------------------------------+
| Allstate  | Binary classification | `link <https://www.kaggle.com/c/ClaimPredictionChallenge>`__           | 13,184,290  | 4228     | last 1,000,000 samples were used as test set |
+-----------+-----------------------+------------------------------------------------------------------------+-------------+----------+----------------------------------------------+

Environment
^^^^^^^^^^^

We ran all experiments on a single Linux server with the following specifications:

+------------------+-----------------+---------------------+
| OS               | CPU             | Memory              |
+==================+=================+=====================+
| Ubuntu 14.04 LTS | 2 \* E5-2670 v3 | DDR4 2133Mhz, 256GB |
+------------------+-----------------+---------------------+

Baseline
^^^^^^^^

We used `xgboost`_ as a baseline.

Both xgboost and LightGBM were built with OpenMP support.

Settings
^^^^^^^^

We set up total 3 settings for experiments. The parameters of these settings are:

1. xgboost:

   .. code::

       eta = 0.1
       max_depth = 8
       num_round = 500
       nthread = 16
       tree_method = exact
       min_child_weight = 100

2. xgboost\_hist (using histogram based algorithm):

   .. code::

       eta = 0.1
       num_round = 500
       nthread = 16
       tree_method = approx
       min_child_weight = 100
       tree_method = hist
       grow_policy = lossguide
       max_depth = 0
       max_leaves = 255

3. LightGBM:

   .. code::

       learning_rate = 0.1
       num_leaves = 255
       num_trees = 500
       num_threads = 16
       min_data_in_leaf = 0
       min_sum_hessian_in_leaf = 100

xgboost grows trees depth-wise and controls model complexity by ``max_depth``.
LightGBM uses a leaf-wise algorithm instead and controls model complexity by ``num_leaves``.
So we cannot compare them in the exact same model setting. For the tradeoff, we use xgboost with ``max_depth=8``, which will have max number leaves to 255, to compare with LightGBM with ``num_leaves=255``.

Other parameters are default values.

Result
^^^^^^

Speed
'''''

We compared speed using only the training task without any test or metric output. We didn't count the time for IO.

The following table is the comparison of time cost:

+-----------+-----------+---------------+------------------+
| Data      | xgboost   | xgboost\_hist | LightGBM         |
+===========+===========+===============+==================+
| Higgs     | 3794.34 s | 551.898 s     | **238.505513 s** |
+-----------+-----------+---------------+------------------+
| Yahoo LTR | 674.322 s | 265.302 s     | **150.18644 s**  |
+-----------+-----------+---------------+------------------+
| MS LTR    | 1251.27 s | 385.201 s     | **215.320316 s** |
+-----------+-----------+---------------+------------------+
| Expo      | 1607.35 s | 588.253 s     | **138.504179 s** |
+-----------+-----------+---------------+------------------+
| Allstate  | 2867.22 s | 1355.71 s     | **348.084475 s** |
+-----------+-----------+---------------+------------------+

LightGBM ran faster than xgboost on all experiment data sets.

Accuracy
''''''''

We computed all accuracy metrics only on the test data set.

+-----------+-----------------+----------+-------------------+--------------+
| Data      | Metric          | xgboost  | xgboost\_hist     | LightGBM     |
+===========+=================+==========+===================+==============+
| Higgs     | AUC             | 0.839593 | **0.845605**      | 0.845154     |
+-----------+-----------------+----------+-------------------+--------------+
| Yahoo LTR | NDCG\ :sub:`1`  | 0.719748 | 0.720223          | **0.732466** |
|           +-----------------+----------+-------------------+--------------+
|           | NDCG\ :sub:`3`  | 0.717813 | 0.721519          | **0.738048** |
|           +-----------------+----------+-------------------+--------------+
|           | NDCG\ :sub:`5`  | 0.737849 | 0.739904          | **0.756548** |
|           +-----------------+----------+-------------------+--------------+
|           | NDCG\ :sub:`10` | 0.78089  | 0.783013          | **0.796818** |
+-----------+-----------------+----------+-------------------+--------------+
| MS LTR    | NDCG\ :sub:`1`  | 0.483956 | 0.488649          | **0.524255** |
|           +-----------------+----------+-------------------+--------------+
|           | NDCG\ :sub:`3`  | 0.467951 | 0.473184          | **0.505327** |
|           +-----------------+----------+-------------------+--------------+
|           | NDCG\ :sub:`5`  | 0.472476 | 0.477438          | **0.510007** |
|           +-----------------+----------+-------------------+--------------+
|           | NDCG\ :sub:`10` | 0.492429 | 0.496967          | **0.527371** |
+-----------+-----------------+----------+-------------------+--------------+
| Expo      | AUC             | 0.756713 | **0.777777**      | 0.777543     |
+-----------+-----------------+----------+-------------------+--------------+
| Allstate  | AUC             | 0.607201 | 0.609042          | **0.609167** |
+-----------+-----------------+----------+-------------------+--------------+

Memory Consumption
''''''''''''''''''

We monitored RES while running training task. And we set ``two_round=true`` (this will increase data-loading time and
reduce peak memory usage but not affect training speed or accuracy) in LightGBM to reduce peak memory usage.

+-----------+---------+---------------+-------------+
| Data      | xgboost | xgboost\_hist | LightGBM    |
+===========+=========+===============+=============+
| Higgs     | 4.853GB | 3.784GB       | **0.868GB** |
+-----------+---------+---------------+-------------+
| Yahoo LTR | 1.907GB | 1.468GB       | **0.831GB** |
+-----------+---------+---------------+-------------+
| MS LTR    | 5.469GB | 3.654GB       | **0.886GB** |
+-----------+---------+---------------+-------------+
| Expo      | 1.553GB | 1.393GB       | **0.543GB** |
+-----------+---------+---------------+-------------+
| Allstate  | 6.237GB | 4.990GB       | **1.027GB** |
+-----------+---------+---------------+-------------+

Parallel Experiment
-------------------

Data
^^^^

We used a terabyte click log dataset to conduct parallel experiments. Details are listed in following table:

+--------+-----------------------+---------+---------------+----------+
| Data   | Task                  | Link    | #Data         | #Feature |
+========+=======================+=========+===============+==========+
| Criteo | Binary classification | `link`_ | 1,700,000,000 | 67       |
+--------+-----------------------+---------+---------------+----------+

This data contains 13 integer features and 26 categorical features for 24 days of click logs.
We statisticized the clickthrough rate (CTR) and count for these 26 categorical features from the first ten days.
Then we used next ten days' data, after replacing the categorical features by the corresponding CTR and count, as training data.
The processed training data have a total of 1.7 billions records and 67 features.

Environment
^^^^^^^^^^^

We ran our experiments on 16 Windows servers with the following specifications:

+---------------------+-----------------+---------------------+-------------------------------------------+
| OS                  | CPU             | Memory              | Network Adapter                           |
+=====================+=================+=====================+===========================================+
| Windows Server 2012 | 2 \* E5-2670 v2 | DDR3 1600Mhz, 256GB | Mellanox ConnectX-3, 54Gbps, RDMA support |
+---------------------+-----------------+---------------------+-------------------------------------------+

Settings
^^^^^^^^

.. code::

    learning_rate = 0.1
    num_leaves = 255
    num_trees = 100
    num_thread = 16
    tree_learner = data

We used data parallel here because this data is large in ``#data`` but small in ``#feature``. Other parameters were default values.

Results
^^^^^^^

+----------+---------------+---------------------------+
| #Machine | Time per Tree | Memory Usage(per Machine) |
+==========+===============+===========================+
| 1        | 627.8 s       | 176GB                     |
+----------+---------------+---------------------------+
| 2        | 311 s         | 87GB                      |
+----------+---------------+---------------------------+
| 4        | 156 s         | 43GB                      |
+----------+---------------+---------------------------+
| 8        | 80 s          | 22GB                      |
+----------+---------------+---------------------------+
| 16       | 42 s          | 11GB                      |
+----------+---------------+---------------------------+

The results show that LightGBM achieves a linear speedup with parallel learning.

GPU Experiments
---------------

Refer to `GPU Performance <./GPU-Performance.rst>`__.

.. _repo: https://github.com/guolinke/boosting_tree_benchmarks

.. _xgboost: https://github.com/dmlc/xgboost

.. _link: http://labs.criteo.com/2013/12/download-terabyte-click-logs/
