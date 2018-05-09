GPU SDK Correspondence and Device Targeting Table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GPU Targets Table
=================

When using OpenCL SDKs, targeting CPU and GPU at the same time is sometimes possible.
This is especially true for Intel OpenCL SDK and AMD APP SDK.

You can find below a table of correspondence:

+---------------------------+-----------------+-----------------+-----------------+--------------+
| SDK                       | CPU Intel/AMD   | GPU Intel       | GPU AMD         | GPU NVIDIA   |
+===========================+=================+=================+=================+==============+
| `Intel SDK for OpenCL`_   | Supported       | Supported \*    | Supported       | Untested     |
+---------------------------+-----------------+-----------------+-----------------+--------------+
| `AMD APP SDK`_            | Supported       | Untested \*     | Supported       | Fails        |
+---------------------------+-----------------+-----------------+-----------------+--------------+
| `NVIDIA CUDA Toolkit`_    | Fails    \*\*   | Fails    \*\*   | Fails    \*\*   | Supported    |
+---------------------------+-----------------+-----------------+-----------------+--------------+

Legend:

-  \* Not usable directly.
-  \*\* Reported as unsupported in public forums.

AMD GPUs using Intel SDK for OpenCL is not a typo, nor AMD APP SDK compatibility with CPUs.

--------------

Targeting Table
===============

We present the following scenarii:

-  CPU, no GPU
-  Single CPU and GPU (even with integrated graphics)
-  Multiple CPU/GPU

We provide test R code below, but you can use the language of your choice with the examples of your choices:

.. code:: r

    library(lightgbm)
    data(agaricus.train, package = "lightgbm")
    train <- agaricus.train
    train$data[, 1] <- 1:6513
    dtrain <- lgb.Dataset(train$data, label = train$label)
    data(agaricus.test, package = "lightgbm")
    test <- agaricus.test
    dtest <- lgb.Dataset.create.valid(dtrain, test$data, label = test$label)
    valids <- list(test = dtest)

    params <- list(objective = "regression",
                   metric = "rmse",
                   device = "gpu",
                   gpu_platform_id = 0,
                   gpu_device_id = 0,
                   nthread = 1,
                   boost_from_average = FALSE,
                   num_tree_per_iteration = 10,
                   max_bin = 32)
    model <- lgb.train(params,
                       dtrain,
                       2,
                       valids,
                       min_data = 1,
                       learning_rate = 1,
                       early_stopping_rounds = 10)

Using a bad ``gpu_device_id`` is not critical, as it will fallback to:

-  ``gpu_device_id = 0`` if using ``gpu_platform_id = 0``
-  ``gpu_device_id = 1`` if using ``gpu_platform_id = 1``

However, using a bad combination of ``gpu_platform_id`` and ``gpu_device_id`` will lead to a **crash** (you will lose your entire session content).
Beware of it.

Your system might have multiple GPUs from different vendors ("platforms") installed. You can use the `clinfo`_ utility to identify the GPUs on each platform. On Ubuntu, you can install ``clinfo`` by executing ``sudo apt-get install clinfo``. On Windows, you can find a list of your OpenCL devices using the utility `GPUCapsViewer`_. If you have a discrete GPU by AMD/NVIDIA and an integrated GPU by Intel, make sure to select the correct ``gpu_platform_id`` to use the discrete GPU.


CPU Only Architectures
----------------------

When you have a single device (one CPU), OpenCL usage is straightforward: ``gpu_platform_id = 0``, ``gpu_device_id = 0``

This will use the CPU with OpenCL, even though it says it says GPU.

Example:

.. code:: r

    > params <- list(objective = "regression",
    +                metric = "rmse",
    +                device = "gpu",
    +                gpu_platform_id = 0,
    +                gpu_device_id = 0,
    +                nthread = 1,
    +                boost_from_average = FALSE,
    +                num_tree_per_iteration = 10,
    +                max_bin = 32)
    > model <- lgb.train(params,
    +                    dtrain,
    +                    2,
    +                    valids,
    +                    min_data = 1,
    +                    learning_rate = 1,
    +                    early_stopping_rounds = 10)
    [LightGBM] [Info] This is the GPU trainer!!
    [LightGBM] [Info] Total Bins 232
    [LightGBM] [Info] Number of data: 6513, number of used features: 116
    [LightGBM] [Info] Using requested OpenCL platform 0 device 1
    [LightGBM] [Info] Using GPU Device: Intel(R) Core(TM) i7-4600U CPU @ 2.10GHz, Vendor: GenuineIntel
    [LightGBM] [Info] Compiling OpenCL Kernel with 16 bins...
    [LightGBM] [Info] GPU programs have been built
    [LightGBM] [Info] Size of histogram bin entry: 12
    [LightGBM] [Info] 40 dense feature groups (0.12 MB) transfered to GPU in 0.004540 secs. 76 sparse feature groups.
    [LightGBM] [Info] No further splits with positive gain, best gain: -inf
    [LightGBM] [Info] Trained a tree with leaves=16 and max_depth=8
    [1]:    test's rmse:1.10643e-17 
    [LightGBM] [Info] No further splits with positive gain, best gain: -inf
    [LightGBM] [Info] Trained a tree with leaves=7 and max_depth=5
    [2]:    test's rmse:0

Single CPU and GPU (even with integrated graphics)
--------------------------------------------------

If you have integrated graphics card (Intel HD Graphics) and a dedicated graphics card (AMD, NVIDIA),
the dedicated graphics card will automatically override the integrated graphics card.
The workaround is to disable your dedicated graphics card to be able to use your integrated graphics card.

When you have multiple devices (one CPU and one GPU), the order is usually the following:

-  GPU: ``gpu_platform_id = 0``, ``gpu_device_id = 0``,
   sometimes it is usable using ``gpu_platform_id = 1``, ``gpu_device_id = 1`` but at your own risk!

-  CPU: ``gpu_platform_id = 0``, ``gpu_device_id = 1``

Example of GPU (``gpu_platform_id = 0``, ``gpu_device_id = 0``):

.. code:: r

    > params <- list(objective = "regression",
    +                metric = "rmse",
    +                device = "gpu",
    +                gpu_platform_id = 0,
    +                gpu_device_id = 0,
    +                nthread = 1,
    +                boost_from_average = FALSE,
    +                num_tree_per_iteration = 10,
    +                max_bin = 32)
    > model <- lgb.train(params,
    +                    dtrain,
    +                    2,
    +                    valids,
    +                    min_data = 1,
    +                    learning_rate = 1,
    +                    early_stopping_rounds = 10)
    [LightGBM] [Info] This is the GPU trainer!!
    [LightGBM] [Info] Total Bins 232
    [LightGBM] [Info] Number of data: 6513, number of used features: 116
    [LightGBM] [Info] Using GPU Device: Oland, Vendor: Advanced Micro Devices, Inc.
    [LightGBM] [Info] Compiling OpenCL Kernel with 16 bins...
    [LightGBM] [Info] GPU programs have been built
    [LightGBM] [Info] Size of histogram bin entry: 12
    [LightGBM] [Info] 40 dense feature groups (0.12 MB) transfered to GPU in 0.004211 secs. 76 sparse feature groups.
    [LightGBM] [Info] No further splits with positive gain, best gain: -inf
    [LightGBM] [Info] Trained a tree with leaves=16 and max_depth=8
    [1]:    test's rmse:1.10643e-17 
    [LightGBM] [Info] No further splits with positive gain, best gain: -inf
    [LightGBM] [Info] Trained a tree with leaves=7 and max_depth=5
    [2]:    test's rmse:0

Example of CPU (``gpu_platform_id = 0``, ``gpu_device_id = 1``):

.. code:: r

    > params <- list(objective = "regression",
    +                metric = "rmse",
    +                device = "gpu",
    +                gpu_platform_id = 0,
    +                gpu_device_id = 1,
    +                nthread = 1,
    +                boost_from_average = FALSE,
    +                num_tree_per_iteration = 10,
    +                max_bin = 32)
    > model <- lgb.train(params,
    +                    dtrain,
    +                    2,
    +                    valids,
    +                    min_data = 1,
    +                    learning_rate = 1,
    +                    early_stopping_rounds = 10)
    [LightGBM] [Info] This is the GPU trainer!!
    [LightGBM] [Info] Total Bins 232
    [LightGBM] [Info] Number of data: 6513, number of used features: 116
    [LightGBM] [Info] Using requested OpenCL platform 0 device 1
    [LightGBM] [Info] Using GPU Device: Intel(R) Core(TM) i7-4600U CPU @ 2.10GHz, Vendor: GenuineIntel
    [LightGBM] [Info] Compiling OpenCL Kernel with 16 bins...
    [LightGBM] [Info] GPU programs have been built
    [LightGBM] [Info] Size of histogram bin entry: 12
    [LightGBM] [Info] 40 dense feature groups (0.12 MB) transfered to GPU in 0.004540 secs. 76 sparse feature groups.
    [LightGBM] [Info] No further splits with positive gain, best gain: -inf
    [LightGBM] [Info] Trained a tree with leaves=16 and max_depth=8
    [1]:    test's rmse:1.10643e-17 
    [LightGBM] [Info] No further splits with positive gain, best gain: -inf
    [LightGBM] [Info] Trained a tree with leaves=7 and max_depth=5
    [2]:    test's rmse:0

When using a wrong ``gpu_device_id``, it will automatically fallback to ``gpu_device_id = 0``:

.. code:: r

    > params <- list(objective = "regression",
    +                metric = "rmse",
    +                device = "gpu",
    +                gpu_platform_id = 0,
    +                gpu_device_id = 9999,
    +                nthread = 1,
    +                boost_from_average = FALSE,
    +                num_tree_per_iteration = 10,
    +                max_bin = 32)
    > model <- lgb.train(params,
    +                    dtrain,
    +                    2,
    +                    valids,
    +                    min_data = 1,
    +                    learning_rate = 1,
    +                    early_stopping_rounds = 10)
    [LightGBM] [Info] This is the GPU trainer!!
    [LightGBM] [Info] Total Bins 232
    [LightGBM] [Info] Number of data: 6513, number of used features: 116
    [LightGBM] [Info] Using GPU Device: Oland, Vendor: Advanced Micro Devices, Inc.
    [LightGBM] [Info] Compiling OpenCL Kernel with 16 bins...
    [LightGBM] [Info] GPU programs have been built
    [LightGBM] [Info] Size of histogram bin entry: 12
    [LightGBM] [Info] 40 dense feature groups (0.12 MB) transfered to GPU in 0.004211 secs. 76 sparse feature groups.
    [LightGBM] [Info] No further splits with positive gain, best gain: -inf
    [LightGBM] [Info] Trained a tree with leaves=16 and max_depth=8
    [1]:    test's rmse:1.10643e-17 
    [LightGBM] [Info] No further splits with positive gain, best gain: -inf
    [LightGBM] [Info] Trained a tree with leaves=7 and max_depth=5
    [2]:    test's rmse:0

Do not ever run under the following scenario as it is known to crash even if it says it is using the CPU because it is NOT the case:

-  One CPU and one GPU
-  ``gpu_platform_id = 1``, ``gpu_device_id = 0``

.. code:: r

    > params <- list(objective = "regression",
    +                metric = "rmse",
    +                device = "gpu",
    +                gpu_platform_id = 1,
    +                gpu_device_id = 0,
    +                nthread = 1,
    +                boost_from_average = FALSE,
    +                num_tree_per_iteration = 10,
    +                max_bin = 32)
    > model <- lgb.train(params,
    +                    dtrain,
    +                    2,
    +                    valids,
    +                    min_data = 1,
    +                    learning_rate = 1,
    +                    early_stopping_rounds = 10)
    [LightGBM] [Info] This is the GPU trainer!!
    [LightGBM] [Info] Total Bins 232
    [LightGBM] [Info] Number of data: 6513, number of used features: 116
    [LightGBM] [Info] Using requested OpenCL platform 1 device 0
    [LightGBM] [Info] Using GPU Device: Intel(R) Core(TM) i7-4600U CPU @ 2.10GHz, Vendor: Intel(R) Corporation
    [LightGBM] [Info] Compiling OpenCL Kernel with 16 bins...
    terminate called after throwing an instance of 'boost::exception_detail::clone_impl<boost::exception_detail::error_info_injector<boost::compute::opencl_error> >'
      what():  Invalid Program

    This application has requested the Runtime to terminate it in an unusual way.
    Please contact the application's support team for more information.

Multiple CPU and GPU
--------------------

If you have multiple devices (multiple CPUs and multiple GPUs),
you will have to test different ``gpu_device_id`` and different ``gpu_platform_id`` values to find out the values which suits the CPU/GPU you want to use.
Keep in mind that using the integrated graphics card is not directly possible without disabling every dedicated graphics card.

.. _Intel SDK for OpenCL: https://software.intel.com/en-us/articles/opencl-drivers

.. _AMD APP SDK: http://developer.amd.com/  # amd-accelerated-parallel-processing-app-sdk/

.. _NVIDIA CUDA Toolkit: https://developer.nvidia.com/cuda-downloads

.. _clinfo: https://github.com/Oblomov/clinfo

.. _GPUCapsViewer: http://www.ozone3d.net/gpu_caps_viewer/


