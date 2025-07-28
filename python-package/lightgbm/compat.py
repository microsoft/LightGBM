# coding: utf-8
"""Compatibility library."""

from typing import TYPE_CHECKING, Any, List

# scikit-learn is intentionally imported first here,
# see https://github.com/microsoft/LightGBM/issues/6509
"""sklearn"""
try:
    from sklearn import __version__ as _sklearn_version
    from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_sample_weight
    from sklearn.utils.multiclass import check_classification_targets
    from sklearn.utils.validation import assert_all_finite, check_array, check_X_y

    try:
        from sklearn.exceptions import NotFittedError
        from sklearn.model_selection import BaseCrossValidator, GroupKFold, StratifiedKFold
    except ImportError:
        from sklearn.cross_validation import BaseCrossValidator, GroupKFold, StratifiedKFold
        from sklearn.utils.validation import NotFittedError
    try:
        from sklearn.utils.validation import _check_sample_weight
    except ImportError:
        from sklearn.utils.validation import check_consistent_length

        # dummy function to support older version of scikit-learn
        def _check_sample_weight(sample_weight: Any, X: Any, dtype: Any = None) -> Any:
            check_consistent_length(sample_weight, X)
            return sample_weight

    try:
        from sklearn.utils.validation import validate_data
    except ImportError:
        # validate_data() was added in scikit-learn 1.6, this function roughly imitates it for older versions.
        # It can be removed when lightgbm's minimum scikit-learn version is at least 1.6.
        def validate_data(
            _estimator,
            X,
            y="no_validation",
            accept_sparse: bool = True,
            # 'force_all_finite' was renamed to 'ensure_all_finite' in scikit-learn 1.6
            ensure_all_finite: bool = False,
            ensure_min_samples: int = 1,
            # trap other keyword arguments that only work on scikit-learn >=1.6, like 'reset'
            **ignored_kwargs,
        ):
            # it's safe to import _num_features unconditionally because:
            #
            #  * it was first added in scikit-learn 0.24.2
            #  * lightgbm cannot be used with scikit-learn versions older than that
            #  * this validate_data() re-implementation will not be called in scikit-learn>=1.6
            #
            from sklearn.utils.validation import _num_features

            # _num_features() raises a TypeError on 1-dimensional input. That's a problem
            # because scikit-learn's 'check_fit1d' estimator check sets that expectation that
            # estimators must raise a ValueError when a 1-dimensional input is passed to fit().
            #
            # So here, lightgbm avoids calling _num_features() on 1-dimensional inputs.
            if hasattr(X, "shape") and len(X.shape) == 1:
                n_features_in_ = 1
            else:
                n_features_in_ = _num_features(X)

            no_val_y = isinstance(y, str) and y == "no_validation"

            # NOTE: check_X_y() calls check_array() internally, so only need to call one or the other of them here
            if no_val_y:
                X = check_array(
                    X,
                    accept_sparse=accept_sparse,
                    force_all_finite=ensure_all_finite,
                    ensure_min_samples=ensure_min_samples,
                )
            else:
                X, y = check_X_y(
                    X,
                    y,
                    accept_sparse=accept_sparse,
                    force_all_finite=ensure_all_finite,
                    ensure_min_samples=ensure_min_samples,
                )

                # this only needs to be updated at fit() time
                _estimator.n_features_in_ = n_features_in_

            # raise the same error that scikit-learn's `validate_data()` does on scikit-learn>=1.6
            if _estimator.__sklearn_is_fitted__() and _estimator._n_features != n_features_in_:
                raise ValueError(
                    f"X has {n_features_in_} features, but {_estimator.__class__.__name__} "
                    f"is expecting {_estimator._n_features} features as input."
                )

            if no_val_y:
                return X
            else:
                return X, y

    SKLEARN_INSTALLED = True
    _LGBMBaseCrossValidator = BaseCrossValidator
    _LGBMModelBase = BaseEstimator
    _LGBMRegressorBase = RegressorMixin
    _LGBMClassifierBase = ClassifierMixin
    _LGBMLabelEncoder = LabelEncoder
    LGBMNotFittedError = NotFittedError
    _LGBMStratifiedKFold = StratifiedKFold
    _LGBMGroupKFold = GroupKFold
    _LGBMCheckSampleWeight = _check_sample_weight
    _LGBMAssertAllFinite = assert_all_finite
    _LGBMCheckClassificationTargets = check_classification_targets
    _LGBMComputeSampleWeight = compute_sample_weight
    _LGBMValidateData = validate_data
except ImportError:
    SKLEARN_INSTALLED = False

    class _LGBMModelBase:  # type: ignore
        """Dummy class for sklearn.base.BaseEstimator."""

        pass

    class _LGBMClassifierBase:  # type: ignore
        """Dummy class for sklearn.base.ClassifierMixin."""

        pass

    class _LGBMRegressorBase:  # type: ignore
        """Dummy class for sklearn.base.RegressorMixin."""

        pass

    _LGBMBaseCrossValidator = None
    _LGBMLabelEncoder = None
    LGBMNotFittedError = ValueError
    _LGBMStratifiedKFold = None
    _LGBMGroupKFold = None
    _LGBMCheckSampleWeight = None
    _LGBMAssertAllFinite = None
    _LGBMCheckClassificationTargets = None
    _LGBMComputeSampleWeight = None
    _LGBMValidateData = None
    _sklearn_version = None

# additional scikit-learn imports only for type hints
if TYPE_CHECKING:
    # sklearn.utils.Tags can be imported unconditionally once
    # lightgbm's minimum scikit-learn version is 1.6 or higher
    try:
        from sklearn.utils import Tags as _sklearn_Tags
    except ImportError:
        _sklearn_Tags = None


"""pandas"""
try:
    from pandas import DataFrame as pd_DataFrame
    from pandas import Series as pd_Series
    from pandas import concat

    try:
        from pandas import CategoricalDtype as pd_CategoricalDtype
    except ImportError:
        from pandas.api.types import CategoricalDtype as pd_CategoricalDtype
    PANDAS_INSTALLED = True
except ImportError:
    PANDAS_INSTALLED = False

    class pd_Series:  # type: ignore
        """Dummy class for pandas.Series."""

        def __init__(self, *args: Any, **kwargs: Any):
            pass

    class pd_DataFrame:  # type: ignore
        """Dummy class for pandas.DataFrame."""

        def __init__(self, *args: Any, **kwargs: Any):
            pass

    class pd_CategoricalDtype:  # type: ignore
        """Dummy class for pandas.CategoricalDtype."""

        def __init__(self, *args: Any, **kwargs: Any):
            pass

    concat = None

"""matplotlib"""
try:
    import matplotlib  # noqa: F401

    MATPLOTLIB_INSTALLED = True
except ImportError:
    MATPLOTLIB_INSTALLED = False

"""graphviz"""
try:
    import graphviz  # noqa: F401

    GRAPHVIZ_INSTALLED = True
except ImportError:
    GRAPHVIZ_INSTALLED = False

"""datatable"""
try:
    import datatable

    if hasattr(datatable, "Frame"):
        dt_DataTable = datatable.Frame
    else:
        dt_DataTable = datatable.DataTable
    DATATABLE_INSTALLED = True
except ImportError:
    DATATABLE_INSTALLED = False

    class dt_DataTable:  # type: ignore
        """Dummy class for datatable.DataTable."""

        def __init__(self, *args: Any, **kwargs: Any):
            pass


"""dask"""
try:
    import pkg_resources
    pkg_resources.get_distribution('dask')
    DASK_INSTALLED = True
except pkg_resources.DistributionNotFound:
    DASK_INSTALLED = False

