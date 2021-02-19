Distributed Learning Guide
==========================

.. _Parallel Learning Guide:

This guide describes distributed learning in LightGBM. Distributed learning allows the use of multiple machines to produce a single model.

Follow the `Quick Start <./Quick-Start.rst>`__ to know how to use LightGBM first.

How Distributed LightGBM Works
------------------------------

This section describes how distributed learning in LightGBM works. To learn how to do this in various programming languages and frameworks, please see `Integrations <#integrations>`_.

Choose Appropriate Parallel Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

LightGBM provides 3 distributed learning algorithms now.

+--------------------+---------------------------+
| Parallel Algorithm | How to Use                |
+====================+===========================+
| Data parallel      | ``tree_learner=data``     |
+--------------------+---------------------------+
| Feature parallel   | ``tree_learner=feature``  |
+--------------------+---------------------------+
| Voting parallel    | ``tree_learner=voting``   |
+--------------------+---------------------------+

These algorithms are suited for different scenarios, which is listed in the following table:

+-------------------------+-------------------+-----------------+
|                         | #data is small    | #data is large  |
+=========================+===================+=================+
| **#feature is small**   | Feature Parallel  | Data Parallel   |
+-------------------------+-------------------+-----------------+
| **#feature is large**   | Feature Parallel  | Voting Parallel |
+-------------------------+-------------------+-----------------+

More details about these parallel algorithms can be found in `optimization in distributed learning <./Features.rst#optimization-in-distributed-learning>`__.

Integrations
------------

This section describes how to run distributed LightGBM training in various programming languages and frameworks. To learn how distributed learning in LightGBM works generally, please see `How Distributed LightGBM Works <#how-distributed-lightgbm-works>`__.

Apache Spark
^^^^^^^^^^^^

Apache Spark users can use `MMLSpark`_ for machine learning workflows with LightGBM. This project is not maintained by LightGBM's maintainers.

See `this MMLSpark example`_ and the `the MMLSpark documentation`_ for additional information on using LightGBM on Spark.

.. note::

  ``MMLSpark`` is not maintained by LightGBM's maintainers. Bug reports or feature requests should be directed to https://github.com/Azure/mmlspark/issues.

Dask
^^^^

.. versionadded:: 3.2.0

LightGBM's Python package supports distributed learning via `Dask`_. This integration is maintained by LightGBM's maintainers.

Kubeflow
^^^^^^^^

`Kubeflow Fairing`_ supports LightGBM distributed training. `These examples`_ show how to get started with LightGBM and Kubeflow Fairing in a hybrid cloud environment.

Kubeflow users can also use the `Kubeflow XGBoost Operator`_ for machine learning workflows with LightGBM. You can see `this example`_ for more details.

Kubeflow integrations for LightGBM are not maintained by LightGBM's maintainers.

.. note::

  The Kubeflow integrations for LightGBM are not maintained by LightGBM's maintainers. Bug reports or feature requests should be directed to https://github.com/kubeflow/fairing/issues or https://github.com/kubeflow/xgboost-operator/issues.

LightGBM CLI
^^^^^^^^^^^^

.. _Build Parallel Version:

Preparation
'''''''''''

By default, distributed learning with LightGBM uses socket-based communication.

If you need to build parallel version with MPI support, please refer to `Installation Guide <./Installation-Guide.rst#build-mpi-version>`__.

Socket Version
**************

It needs to collect IP of all machines that want to run distributed learning in and allocate one TCP port (assume 12345 here) for all machines,
and change firewall rules to allow income of this port (12345). Then write these IP and ports in one file (assume ``mlist.txt``), like following:

.. code::

    machine1_ip 12345
    machine2_ip 12345

MPI Version
***********

It needs to collect IP (or hostname) of all machines that want to run distributed learning in.
Then write these IP in one file (assume ``mlist.txt``) like following:

.. code::

    machine1_ip
    machine2_ip

**Note**: For Windows users, need to start "smpd" to start MPI service. More details can be found `here`_.

Run Distributed Learning
''''''''''''''''''''''''

.. _Run Parallel Learning:

Socket Version
**************

1. Edit following parameters in config file:

   ``tree_learner=your_parallel_algorithm``, edit ``your_parallel_algorithm`` (e.g. feature/data) here.

   ``num_machines=your_num_machines``, edit ``your_num_machines`` (e.g. 4) here.

   ``machine_list_file=mlist.txt``, ``mlist.txt`` is created in `Preparation section <#preparation>`__.

   ``local_listen_port=12345``, ``12345`` is allocated in `Preparation section <#preparation>`__.

2. Copy data file, executable file, config file and ``mlist.txt`` to all machines.

3. Run following command on all machines, you need to change ``your_config_file`` to real config file.

   For Windows: ``lightgbm.exe config=your_config_file``

   For Linux: ``./lightgbm config=your_config_file``

MPI Version
***********

1. Edit following parameters in config file:

   ``tree_learner=your_parallel_algorithm``, edit ``your_parallel_algorithm`` (e.g. feature/data) here.

   ``num_machines=your_num_machines``, edit ``your_num_machines`` (e.g. 4) here.

2. Copy data file, executable file, config file and ``mlist.txt`` to all machines.

   **Note**: MPI needs to be run in the **same path on all machines**.

3. Run following command on one machine (not need to run on all machines), need to change ``your_config_file`` to real config file.

   For Windows:
   
   .. code::

       mpiexec.exe /machinefile mlist.txt lightgbm.exe config=your_config_file

   For Linux:

   .. code::

       mpiexec --machinefile mlist.txt ./lightgbm config=your_config_file

Example
'''''''

-  `A simple distributed learning example`_

.. _Dask: https://docs.dask.org/en/latest/

.. _MMLSpark: https://aka.ms/spark

.. _this MMLSpark example: https://github.com/Azure/mmlspark/blob/master/notebooks/samples/LightGBM%20-%20Quantile%20Regression%20for%20Drug%20Discovery.ipynb

.. _the MMLSpark Documentation: https://github.com/Azure/mmlspark/blob/master/docs/lightgbm.md

.. _Kubeflow Fairing: https://www.kubeflow.org/docs/components/fairing/fairing-overview

.. _These examples: https://github.com/kubeflow/fairing/tree/master/examples/lightgbm

.. _Kubeflow XGBoost Operator: https://github.com/kubeflow/xgboost-operator

.. _this example: https://github.com/kubeflow/xgboost-operator/tree/master/config/samples/lightgbm-dist

.. _here: https://www.youtube.com/watch?v=iqzXhp5TxUY

.. _A simple distributed learning example: https://github.com/microsoft/lightgbm/tree/master/examples/parallel_learning
