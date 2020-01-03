Python API
==========

.. currentmodule:: lightgbm

Data Structure API
------------------

.. autosummary::
    :toctree: pythonapi/

    Dataset
    Booster

Training API
------------

.. autosummary::
    :toctree: pythonapi/

    train
    cv

Scikit-learn API
----------------

.. warning::

    The last supported version of scikit-learn is ``0.21.3``. Our estimators are incompatible with newer versions.

.. autosummary::
    :toctree: pythonapi/

    LGBMModel
    LGBMClassifier
    LGBMRegressor
    LGBMRanker

Callbacks
---------

.. autosummary::
    :toctree: pythonapi/

    early_stopping
    print_evaluation
    record_evaluation
    reset_parameter

Plotting
--------

.. autosummary::
    :toctree: pythonapi/

    plot_importance
    plot_split_value_histogram
    plot_metric
    plot_tree
    create_tree_digraph
