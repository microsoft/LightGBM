Features
========

This is a short introduction for the features and algorithms used in LightGBM\ `[1] <#references>`__.

This page doesn't contain detailed algorithms, please refer to cited papers or source code if you are interested.

Optimization in Speed and Memory Usage
--------------------------------------

Many boosting tools use pre-sorted based algorithms\ `[2, 3] <#references>`__ (e.g. default algorithm in xgboost) for decision tree learning. It is a simple solution, but not easy to optimize.

LightGBM uses the histogram based algorithms\ `[4, 5, 6] <#references>`__, which bucketing continuous feature(attribute) values into discrete bins, to speed up training procedure and reduce memory usage.
Following are advantages for histogram based algorithms:

-  **Reduce calculation cost of split gain**

   -  Pre-sorted based algorithms need ``O(#data)`` times calculation

   -  Histogram based algorithms only need to calculate ``O(#bins)`` times, and ``#bins`` is far smaller than ``#data``

      -  It still needs ``O(#data)`` times to construct histogram, which only contain sum-up operation

-  **Use histogram subtraction for further speed-up**

   -  To get one leaf's histograms in a binary tree, can use the histogram subtraction of its parent and its neighbor

   -  So it only need to construct histograms for one leaf (with smaller ``#data`` than its neighbor), then can get histograms of its neighbor by histogram subtraction with small cost (``O(#bins)``)
   
-  **Reduce memory usage**

   -  Can replace continuous values to discrete bins. If ``#bins`` is small, can use small data type, e.g. uint8\_t, to store training data

   -  No need to store additional information for pre-sorting feature values

-  **Reduce communication cost for parallel learning**

Sparse Optimization
-------------------

-  Only need ``O(2 * #non_zero_data)`` to construct histogram for sparse features

Optimization in Accuracy
------------------------

Leaf-wise (Best-first) Tree Growth
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most decision tree learning algorithms grow tree by level (depth)-wise, like the following image:

.. image:: ./_static/images/level-wise.png
   :align: center

LightGBM grows tree by leaf-wise (best-first)\ `[7] <#references>`__. It will choose the leaf with max delta loss to grow.
When growing same ``#leaf``, leaf-wise algorithm can reduce more loss than level-wise algorithm.

Leaf-wise may cause over-fitting when ``#data`` is small.
So, LightGBM can use an additional parameter ``max_depth`` to limit depth of tree and avoid over-fitting (tree still grows by leaf-wise).

.. image:: ./_static/images/leaf-wise.png
   :align: center

Optimal Split for Categorical Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We often convert the categorical features into one-hot coding.
However, it is not a good solution in tree learner.
The reason is, for the high cardinality categorical features, it will grow the very unbalance tree, and needs to grow very deep to achieve the good accuracy.

Actually, the optimal solution is partitioning the categorical feature into 2 subsets, and there are ``2^(k-1) - 1`` possible partitions.
But there is a efficient solution for regression tree\ `[8] <#references>`__. It needs about ``k * log(k)`` to find the optimal partition.

The basic idea is reordering the categories according to the relevance of training target.
More specifically, reordering the histogram (of categorical feature) according to it's accumulate values (``sum_gradient / sum_hessian``), then find the best split on the sorted histogram.

Optimization in Network Communication
-------------------------------------

It only needs to use some collective communication algorithms, like "All reduce", "All gather" and "Reduce scatter", in parallel learning of LightGBM.
LightGBM implement state-of-art algorithms\ `[9] <#references>`__.
These collective communication algorithms can provide much better performance than point-to-point communication.

Optimization in Parallel Learning
---------------------------------

LightGBM provides following parallel learning algorithms.

Feature Parallel
~~~~~~~~~~~~~~~~

Traditional Algorithm
^^^^^^^^^^^^^^^^^^^^^

Feature parallel aims to parallel the "Find Best Split" in the decision tree. The procedure of traditional feature parallel is:

1. Partition data vertically (different machines have different feature set)

2. Workers find local best split point {feature, threshold} on local feature set

3. Communicate local best splits with each other and get the best one

4. Worker with best split to perform split, then send the split result of data to other workers

5. Other workers split data according received data

The shortage of traditional feature parallel:

-  Has computation overhead, since it cannot speed up "split", whose time complexity is ``O(#data)``.
   Thus, feature parallel cannot speed up well when ``#data`` is large.

-  Need communication of split result, which cost about ``O(#data / 8)`` (one bit for one data).

Feature Parallel in LightGBM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Since feature parallel cannot speed up well when ``#data`` is large, we make a little change here: instead of partitioning data vertically, every worker holds the full data.
Thus, LightGBM doesn't need to communicate for split result of data since every worker know how to split data.
And ``#data`` won't be larger, so it is reasonable to hold full data in every machine.

The procedure of feature parallel in LightGBM:

1. Workers find local best split point {feature, threshold} on local feature set

2. Communicate local best splits with each other and get the best one

3. Perform best split

However, this feature parallel algorithm still suffers from computation overhead for "split" when ``#data`` is large.
So it will be better to use data parallel when ``#data`` is large.

Data Parallel
~~~~~~~~~~~~~

Traditional Algorithm
^^^^^^^^^^^^^^^^^^^^^

Data parallel aims to parallel the whole decision learning. The procedure of data parallel is:

1. Partition data horizontally

2. Workers use local data to construct local histograms

3. Merge global histograms from all local histograms

4. Find best split from merged global histograms, then perform splits

The shortage of traditional data parallel:

-  High communication cost.
   If using point-to-point communication algorithm, communication cost for one machine is about ``O(#machine * #feature * #bin)``.
   If using collective communication algorithm (e.g. "All Reduce"), communication cost is about ``O(2 * #feature * #bin)`` (check cost of "All Reduce" in chapter 4.5 at `[9] <#references>`__).

Data Parallel in LightGBM
^^^^^^^^^^^^^^^^^^^^^^^^^

We reduce communication cost of data parallel in LightGBM:

1. Instead of "Merge global histograms from all local histograms", LightGBM use "Reduce Scatter" to merge histograms of different (non-overlapping) features for different workers.
   Then workers find local best split on local merged histograms and sync up global best split.

2. As aforementioned, LightGBM use histogram subtraction to speed up training.
   Based on this, we can communicate histograms only for one leaf, and get its neighbor's histograms by subtraction as well.

Above all, we reduce communication cost to ``O(0.5 * #feature * #bin)`` for data parallel in LightGBM.

Voting Parallel
~~~~~~~~~~~~~~~

Voting parallel further reduce the communication cost in `Data Parallel <#data-parallel>`__ to constant cost.
It uses two stage voting to reduce the communication cost of feature histograms\ `[10] <#references>`__.

GPU Support
-----------

Thanks `@huanzhang12 <https://github.com/huanzhang12>`__ for contributing this feature. Please read `[11] <#references>`__ to get more details.

- `GPU Installation <./Installation-Guide.rst#build-gpu-version>`__

- `GPU Tutorial <./GPU-Tutorial.rst>`__

Applications and Metrics
------------------------

Support following application:

-  regression, the objective function is L2 loss

-  binary classification, the objective function is logloss

-  multi classification

-  cross-entropy

-  lambdarank, the objective function is lambdarank with NDCG

Support following metrics:

-  L1 loss

-  L2 loss

-  Log loss

-  Classification error rate

-  AUC

-  NDCG

-  MAP

-  Multi class log loss

-  Multi class error rate

-  Fair

-  Huber

-  Poisson

-  Quantile

-  MAPE

-  Kullback-Leibler

-  Gamma

-  Tweedie

For more details, please refer to `Parameters <./Parameters.rst#metric-parameters>`__.

Other Features
--------------

-  Limit ``max_depth`` of tree while grows tree leaf-wise

-  `DART <https://arxiv.org/abs/1505.01866>`__

-  L1/L2 regularization

-  Bagging

-  Column(feature) sub-sample

-  Continued train with input GBDT model

-  Continued train with the input score file

-  Weighted training

-  Validation metric output during training

-  Multi validation data

-  Multi metrics

-  Early stopping (both training and prediction)

-  Prediction for leaf index

For more details, please refer to `Parameters <./Parameters.rst>`__.

References
----------

[1] Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu. "`LightGBM\: A Highly Efficient Gradient Boosting Decision Tree`_." In Advances in Neural Information Processing Systems (NIPS), pp. 3149-3157. 2017.

[2] Mehta, Manish, Rakesh Agrawal, and Jorma Rissanen. "SLIQ: A fast scalable classifier for data mining." International Conference on Extending Database Technology. Springer Berlin Heidelberg, 1996.

[3] Shafer, John, Rakesh Agrawal, and Manish Mehta. "SPRINT: A scalable parallel classifier for data mining." Proc. 1996 Int. Conf. Very Large Data Bases. 1996.

[4] Ranka, Sanjay, and V. Singh. "CLOUDS: A decision tree classifier for large datasets." Proceedings of the 4th Knowledge Discovery and Data Mining Conference. 1998.

[5] Machado, F. P. "Communication and memory efficient parallel decision tree construction." (2003).

[6] Li, Ping, Qiang Wu, and Christopher J. Burges. "Mcrank: Learning to rank using multiple classification and gradient boosting." Advances in neural information processing systems. 2007.

[7] Shi, Haijian. "Best-first decision tree learning." Diss. The University of Waikato, 2007.

[8] Walter D. Fisher. "`On Grouping for Maximum Homogeneity`_." Journal of the American Statistical Association. Vol. 53, No. 284 (Dec., 1958), pp. 789-798.

[9] Thakur, Rajeev, Rolf Rabenseifner, and William Gropp. "`Optimization of collective communication operations in MPICH`_." International Journal of High Performance Computing Applications 19.1 (2005): 49-66.

[10] Qi Meng, Guolin Ke, Taifeng Wang, Wei Chen, Qiwei Ye, Zhi-Ming Ma, Tieyan Liu. "`A Communication-Efficient Parallel Algorithm for Decision Tree`_." Advances in Neural Information Processing Systems 29 (NIPS 2016).

[11] Huan Zhang, Si Si and Cho-Jui Hsieh. "`GPU Acceleration for Large-scale Tree Boosting`_." arXiv:1706.08359, 2017.

.. _LightGBM\: A Highly Efficient Gradient Boosting Decision Tree: https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf

.. _On Grouping for Maximum Homogeneity: https://www.researchgate.net/publication/242580910_On_Grouping_for_Maximum_Homogeneity

.. _Optimization of collective communication operations in MPICH: http://wwwi10.lrr.in.tum.de/~gerndt/home/Teaching/HPCSeminar/mpich_multi_coll.pdf

.. _A Communication-Efficient Parallel Algorithm for Decision Tree: http://papers.nips.cc/paper/6381-a-communication-efficient-parallel-algorithm-for-decision-tree

.. _GPU Acceleration for Large-scale Tree Boosting: https://arxiv.org/abs/1706.08359
