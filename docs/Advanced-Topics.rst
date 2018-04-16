Advanced Topics
===============

Missing Value Handle
--------------------

-  LightGBM enables the missing value handle by default, you can disable it by set ``use_missing=false``.

-  LightGBM uses NA (NaN) to represent the missing value by default, you can change it to use zero by set ``zero_as_missing=true``.

-  When ``zero_as_missing=false`` (default), the unshown value in sparse matrices (and LightSVM) is treated as zeros.

-  When ``zero_as_missing=true``, NA and zeros (including unshown value in sparse matrices (and LightSVM)) are treated as missing.

Categorical Feature Support
---------------------------

-  LightGBM can offer a good accuracy when using native categorical features. Not like simply one-hot coding, LightGBM can find the optimal split of categorical features.
   Such an optimal split can provide the much better accuracy than one-hot coding solution.

-  Use ``categorical_feature`` to specify the categorical features.
   Refer to the parameter ``categorical_feature`` in `Parameters <./Parameters.rst>`__.

-  Converting to ``int`` type is needed first, and there is support for non-negative numbers only.
   It is better to convert into continues ranges.

-  Use ``min_data_per_group``, ``cat_smooth`` to deal with over-fitting
   (when ``#data`` is small or ``#category`` is large).

-  For categorical features with high cardinality (``#category`` is large), it is better to convert it to numerical features.

LambdaRank
----------

-  The label should be ``int`` type, and larger numbers represent the higher relevance (e.g. 0:bad, 1:fair, 2:good, 3:perfect).

-  Use ``label_gain`` to set the gain(weight) of ``int`` label.

-  Use ``max_position`` to set the NDCG optimization position.

Parameters Tuning
-----------------

-  Refer to `Parameters Tuning <./Parameters-Tuning.rst>`__.

Parallel Learning
-----------------

-  Refer to `Parallel Learning Guide <./Parallel-Learning-Guide.rst>`__.

GPU Support
-----------

-  Refer to `GPU Tutorial <./GPU-Tutorial.rst>`__ and `GPU Targets <./GPU-Targets.rst>`__.

Recommendations for gcc Users (MinGW, \*nix)
--------------------------------------------

-  Refer to `gcc Tips <./gcc-Tips.rst>`__.
