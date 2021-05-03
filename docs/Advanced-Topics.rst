Advanced Topics
===============

Missing Value Handle
--------------------

-  LightGBM enables the missing value handle by default. Disable it by setting ``use_missing=false``.

-  LightGBM uses NA (NaN) to represent missing values by default. Change it to use zero by setting ``zero_as_missing=true``.

-  When ``zero_as_missing=false`` (default), the unrecorded values in sparse matrices (and LightSVM) are treated as zeros.

-  When ``zero_as_missing=true``, NA and zeros (including unrecorded values in sparse matrices (and LightSVM)) are treated as missing.

Categorical Feature Support
---------------------------

-  LightGBM offers good accuracy with integer-encoded categorical features. LightGBM applies
   `Fisher (1958) <https://www.tandfonline.com/doi/abs/10.1080/01621459.1958.10501479>`_
   to find the optimal split over categories as
   `described here <./Features.rst#optimal-split-for-categorical-features>`_. This often performs better than one-hot encoding.

-  Use ``categorical_feature`` to specify the categorical features.
   Refer to the parameter ``categorical_feature`` in `Parameters <./Parameters.rst#categorical_feature>`__.

-  Categorical features must be encoded as non-negative integers (``int``) less than ``Int32.MaxValue`` (2147483647).
   It is best to use a contiguous range of integers started from zero.

-  Use ``min_data_per_group``, ``cat_smooth`` to deal with over-fitting (when ``#data`` is small or ``#category`` is large).

-  For a categorical feature with high cardinality (``#category`` is large), it often works best to
   treat the feature as numeric, either by simply ignoring the categorical interpretation of the integers or
   by embedding the categories in a low-dimensional numeric space.

LambdaRank
----------

-  The label should be of type ``int``, such that larger numbers correspond to higher relevance (e.g. 0:bad, 1:fair, 2:good, 3:perfect).

-  Use ``label_gain`` to set the gain(weight) of ``int`` label.

-  Use ``lambdarank_truncation_level`` to truncate the max DCG.

Cost Efficient Gradient Boosting
--------------------------------

`Cost Efficient Gradient Boosting <https://papers.nips.cc/paper/6753-cost-efficient-gradient-boosting.pdf>`_ (CEGB)  makes it possible to penalise boosting based on the cost of obtaining feature values.
CEGB penalises learning in the following ways:

- Each time a tree is split, a penalty of ``cegb_penalty_split`` is applied.
- When a feature is used for the first time, ``cegb_penalty_feature_coupled`` is applied. This penalty can be different for each feature and should be specified as one ``double`` per feature.
- When a feature is used for the first time for a data row, ``cegb_penalty_feature_lazy`` is applied. Like ``cegb_penalty_feature_coupled``, this penalty is specified as one ``double`` per feature.

Each of the penalties above is scaled by ``cegb_tradeoff``.
Using this parameter, it is possible to change the overall strength of the CEGB penalties by changing only one parameter.

Parameters Tuning
-----------------

-  Refer to `Parameters Tuning <./Parameters-Tuning.rst>`__.

.. _Parallel Learning:

Distributed Learning
--------------------

-  Refer to `Distributed Learning Guide <./Parallel-Learning-Guide.rst>`__.

GPU Support
-----------

-  Refer to `GPU Tutorial <./GPU-Tutorial.rst>`__ and `GPU Targets <./GPU-Targets.rst>`__.

Recommendations for gcc Users (MinGW, \*nix)
--------------------------------------------

-  Refer to `gcc Tips <./gcc-Tips.rst>`__.
