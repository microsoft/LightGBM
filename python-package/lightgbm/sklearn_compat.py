"""Ease developer experience to support multiple versions of scikit-learn.

This file is intended to be vendored in your project if you do not want to depend on
`sklearn-compat` as a package. Then, you can import directly from this file.

Be aware that depending on `sklearn-compat` does not add any additional dependencies:
we are only depending on `scikit-learn`.

Version: 0.1.0
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from typing import Callable, Literal

import sklearn
from sklearn.utils._param_validation import validate_parameter_constraints
from sklearn.utils.fixes import parse_version

sklearn_version = parse_version(parse_version(sklearn.__version__).base_version)


########################################################################################
# The following code does not depend on the sklearn version
########################################################################################


# parameters validation
class ParamsValidationMixin:
    """Mixin class to validate parameters."""

    def _validate_params(self):
        """Validate types and values of constructor parameters.

        The expected type and values must be defined in the `_parameter_constraints`
        class attribute, which is a dictionary `param_name: list of constraints`. See
        the docstring of `validate_parameter_constraints` for a description of the
        accepted constraints.
        """
        if hasattr(self, "_parameter_constraints"):
            validate_parameter_constraints(
                self._parameter_constraints,
                self.get_params(deep=False),
                caller_name=self.__class__.__name__,
            )


# tags infrastructure
def _dataclass_args():
    if sys.version_info < (3, 10):
        return {}
    return {"slots": True}


def get_tags(estimator):
    """Get estimator tags in a consistent format across different sklearn versions.

    This function provides compatibility between sklearn versions before and after 1.6.
    It returns either a Tags object (sklearn >= 1.6) or a converted Tags object from
    the dictionary format (sklearn < 1.6) containing metadata about the estimator's
    requirements and capabilities.

    Parameters
    ----------
    estimator : estimator object
        A scikit-learn estimator instance.

    Returns
    -------
    tags : Tags
        An object containing metadata about the estimator's requirements and
        capabilities (e.g., input types, fitting requirements, classifier/regressor
        specific tags).
    """
    try:
        from sklearn.utils._tags import get_tags

        return get_tags(estimator)
    except ImportError:
        from sklearn.utils._tags import _safe_tags

        return _to_new_tags(_safe_tags(estimator), estimator)


def _to_new_tags(old_tags, estimator=None):
    """Utility function convert old tags (dictionary) to new tags (dataclass)."""
    input_tags = InputTags(
        one_d_array="1darray" in old_tags["X_types"],
        two_d_array="2darray" in old_tags["X_types"],
        three_d_array="3darray" in old_tags["X_types"],
        sparse="sparse" in old_tags["X_types"],
        categorical="categorical" in old_tags["X_types"],
        string="string" in old_tags["X_types"],
        dict="dict" in old_tags["X_types"],
        positive_only=old_tags["requires_positive_X"],
        allow_nan=old_tags["allow_nan"],
        pairwise=old_tags["pairwise"],
    )
    target_tags = TargetTags(
        required=old_tags["requires_y"],
        one_d_labels="1dlabels" in old_tags["X_types"],
        two_d_labels="2dlabels" in old_tags["X_types"],
        positive_only=old_tags["requires_positive_y"],
        multi_output=old_tags["multioutput"] or old_tags["multioutput_only"],
        single_output=not old_tags["multioutput_only"],
    )
    if estimator is not None and (
        hasattr(estimator, "transform") or hasattr(estimator, "fit_transform")
    ):
        transformer_tags = TransformerTags(
            preserves_dtype=old_tags["preserves_dtype"],
        )
    else:
        transformer_tags = None
    estimator_type = getattr(estimator, "_estimator_type", None)
    if estimator_type == "classifier":
        classifier_tags = ClassifierTags(
            poor_score=old_tags["poor_score"],
            multi_class=not old_tags["binary_only"],
            multi_label=old_tags["multilabel"],
        )
    else:
        classifier_tags = None
    if estimator_type == "regressor":
        regressor_tags = RegressorTags(
            poor_score=old_tags["poor_score"],
            multi_label=old_tags["multilabel"],
        )
    else:
        regressor_tags = None
    return Tags(
        estimator_type=estimator_type,
        target_tags=target_tags,
        transformer_tags=transformer_tags,
        classifier_tags=classifier_tags,
        regressor_tags=regressor_tags,
        input_tags=input_tags,
        array_api_support=old_tags["array_api_support"],
        no_validation=old_tags["no_validation"],
        non_deterministic=old_tags["non_deterministic"],
        requires_fit=old_tags["requires_fit"],
        _skip_test=old_tags["_skip_test"],
    )


########################################################################################
# Upgrading for scikit-learn 1.4
########################################################################################


if sklearn_version < parse_version("1.4"):

    def _is_fitted(estimator, attributes=None, all_or_any=all):
        """Determine if an estimator is fitted

        Parameters
        ----------
        estimator : estimator instance
            Estimator instance for which the check is performed.

        attributes : str, list or tuple of str, default=None
            Attribute name(s) given as string or a list/tuple of strings
            Eg.: ``["coef_", "estimator_", ...], "coef_"``

            If `None`, `estimator` is considered fitted if there exist an
            attribute that ends with a underscore and does not start with double
            underscore.

        all_or_any : callable, {all, any}, default=all
            Specify whether all or any of the given attributes must exist.

        Returns
        -------
        fitted : bool
            Whether the estimator is fitted.
        """
        if attributes is not None:
            if not isinstance(attributes, (list, tuple)):
                attributes = [attributes]
            return all_or_any([hasattr(estimator, attr) for attr in attributes])

        if hasattr(estimator, "__sklearn_is_fitted__"):
            return estimator.__sklearn_is_fitted__()

        fitted_attrs = [
            v for v in vars(estimator) if v.endswith("_") and not v.startswith("__")
        ]
        return len(fitted_attrs) > 0

else:
    from sklearn.utils.validation import _is_fitted  # noqa: F401


########################################################################################
# Upgrading for scikit-learn 1.5
########################################################################################


if sklearn_version < parse_version("1.5"):
    # chunking
    # extmath
    # fixes
    from sklearn.utils import (
        _IS_32BIT,  # noqa: F401
        _approximate_mode,  # noqa: F401
        _in_unstable_openblas_configuration,  # noqa: F401
        gen_batches,  # noqa: F401
        gen_even_slices,  # noqa: F401
        get_chunk_n_rows,  # noqa: F401
        safe_sqr,  # noqa: F401
    )
    from sklearn.utils import _chunk_generator as chunk_generator  # noqa: F401

    _IS_WASM = platform.machine() in ["wasm32", "wasm64"]
    # indexing
    # mask
    # missing
    # optional dependencies
    # user interface
    # validation
    from sklearn.utils import (
        _determine_key_type,  # noqa: F401
        _get_column_indices,  # noqa: F401
        _print_elapsed_time,  # noqa: F401
        _safe_assign,  # noqa: F401
        _safe_indexing,  # noqa: F401
        _to_object_array,  # noqa: F401
        axis0_safe_slice,  # noqa: F401
        check_matplotlib_support,  # noqa: F401
        check_pandas_support,  # noqa: F401
        indices_to_mask,  # noqa: F401
        is_scalar_nan,  # noqa: F401
        resample,  # noqa: F401
        safe_mask,  # noqa: F401
        shuffle,  # noqa: F401
    )
    from sklearn.utils import _is_pandas_na as is_pandas_na  # noqa: F401
else:
    # chunking
    from sklearn.utils._chunking import (
        chunk_generator,  # noqa: F401
        gen_batches,  # noqa: F401
        gen_even_slices,  # noqa: F401
        get_chunk_n_rows,  # noqa: F401
    )

    # indexing
    from sklearn.utils._indexing import (
        _determine_key_type,  # noqa: F401
        _get_column_indices,  # noqa: F401
        _safe_assign,  # noqa: F401
        _safe_indexing,  # noqa: F401
        resample,  # noqa: F401
        shuffle,  # noqa: F401
    )

    # mask
    from sklearn.utils._mask import (
        axis0_safe_slice,  # noqa: F401
        indices_to_mask,  # noqa: F401
        safe_mask,  # noqa: F401
    )

    # missing
    from sklearn.utils._missing import (
        is_pandas_na,  # noqa: F401
        is_scalar_nan,  # noqa: F401
    )

    # optional dependencies
    from sklearn.utils._optional_dependencies import (  # noqa: F401
        check_matplotlib_support,
        check_pandas_support,  # noqa: F401
    )

    # user interface
    from sklearn.utils._user_interface import _print_elapsed_time  # noqa: F401

    # extmath
    from sklearn.utils.extmath import (
        _approximate_mode,  # noqa: F401
        safe_sqr,  # noqa: F401
    )

    # fixes
    from sklearn.utils.fixes import (
        _IS_32BIT,  # noqa: F401
        _IS_WASM,  # noqa: F401
        _in_unstable_openblas_configuration,  # noqa: F401
    )

    # validation
    from sklearn.utils.validation import _to_object_array  # noqa: F401

########################################################################################
# Upgrading for scikit-learn 1.6
########################################################################################


if sklearn_version < parse_version("1.6"):
    # test_common
    from sklearn.utils.estimator_checks import _construct_instance

    def type_of_target(y, input_name="", *, raise_unknown=False):
        # fix for raise_unknown which is introduced in scikit-learn 1.6
        from sklearn.utils.multiclass import type_of_target

        def _raise_or_return(target_type):
            """Depending on the value of raise_unknown, either raise an error or
            return 'unknown'.
            """
            if raise_unknown and target_type == "unknown":
                input = input_name if input_name else "data"
                raise ValueError(f"Unknown label type for {input}: {y!r}")
            else:
                return target_type

        target_type = type_of_target(y, input_name=input_name)
        return _raise_or_return(target_type)

    def _construct_instances(Estimator):
        yield _construct_instance(Estimator)

    # validation
    def validate_data(_estimator, /, **kwargs):
        if "ensure_all_finite" in kwargs:
            force_all_finite = kwargs.pop("ensure_all_finite")
        else:
            force_all_finite = True
        return _estimator._validate_data(**kwargs, force_all_finite=force_all_finite)

    def _check_n_features(estimator, X, *, reset):
        return estimator._check_n_features(X, reset=reset)

    def _check_feature_names(estimator, X, *, reset):
        return estimator._check_feature_names(X, reset=reset)

    # tags infrastructure
    @dataclass(**_dataclass_args())
    class InputTags:
        """Tags for the input data.

        Parameters
        ----------
        one_d_array : bool, default=False
            Whether the input can be a 1D array.

        two_d_array : bool, default=True
            Whether the input can be a 2D array. Note that most common
            tests currently run only if this flag is set to ``True``.

        three_d_array : bool, default=False
            Whether the input can be a 3D array.

        sparse : bool, default=False
            Whether the input can be a sparse matrix.

        categorical : bool, default=False
            Whether the input can be categorical.

        string : bool, default=False
            Whether the input can be an array-like of strings.

        dict : bool, default=False
            Whether the input can be a dictionary.

        positive_only : bool, default=False
            Whether the estimator requires positive X.

        allow_nan : bool, default=False
            Whether the estimator supports data with missing values encoded as `np.nan`.

        pairwise : bool, default=False
            This boolean attribute indicates whether the data (`X`),
            :term:`fit` and similar methods consists of pairwise measures
            over samples rather than a feature representation for each
            sample.  It is usually `True` where an estimator has a
            `metric` or `affinity` or `kernel` parameter with value
            'precomputed'. Its primary purpose is to support a
            :term:`meta-estimator` or a cross validation procedure that
            extracts a sub-sample of data intended for a pairwise
            estimator, where the data needs to be indexed on both axes.
            Specifically, this tag is used by
            `sklearn.utils.metaestimators._safe_split` to slice rows and
            columns.
        """

        one_d_array: bool = False
        two_d_array: bool = True
        three_d_array: bool = False
        sparse: bool = False
        categorical: bool = False
        string: bool = False
        dict: bool = False
        positive_only: bool = False
        allow_nan: bool = False
        pairwise: bool = False

    @dataclass(**_dataclass_args())
    class TargetTags:
        """Tags for the target data.

        Parameters
        ----------
        required : bool
            Whether the estimator requires y to be passed to `fit`,
            `fit_predict` or `fit_transform` methods. The tag is ``True``
            for estimators inheriting from `~sklearn.base.RegressorMixin`
            and `~sklearn.base.ClassifierMixin`.

        one_d_labels : bool, default=False
            Whether the input is a 1D labels (y).

        two_d_labels : bool, default=False
            Whether the input is a 2D labels (y).

        positive_only : bool, default=False
            Whether the estimator requires a positive y (only applicable
            for regression).

        multi_output : bool, default=False
            Whether a regressor supports multi-target outputs or a classifier supports
            multi-class multi-output.

        single_output : bool, default=True
            Whether the target can be single-output. This can be ``False`` if the
            estimator supports only multi-output cases.
        """

        required: bool
        one_d_labels: bool = False
        two_d_labels: bool = False
        positive_only: bool = False
        multi_output: bool = False
        single_output: bool = True

    @dataclass(**_dataclass_args())
    class TransformerTags:
        """Tags for the transformer.

        Parameters
        ----------
        preserves_dtype : list[str], default=["float64"]
            Applies only on transformers. It corresponds to the data types
            which will be preserved such that `X_trans.dtype` is the same
            as `X.dtype` after calling `transformer.transform(X)`. If this
            list is empty, then the transformer is not expected to
            preserve the data type. The first value in the list is
            considered as the default data type, corresponding to the data
            type of the output when the input data type is not going to be
            preserved.
        """

        preserves_dtype: list[str] = field(default_factory=lambda: ["float64"])

    @dataclass(**_dataclass_args())
    class ClassifierTags:
        """Tags for the classifier.

        Parameters
        ----------
        poor_score : bool, default=False
            Whether the estimator fails to provide a "reasonable" test-set
            score, which currently for classification is an accuracy of
            0.83 on ``make_blobs(n_samples=300, random_state=0)``. The
            datasets and values are based on current estimators in scikit-learn
            and might be replaced by something more systematic.

        multi_class : bool, default=True
            Whether the classifier can handle multi-class
            classification. Note that all classifiers support binary
            classification. Therefore this flag indicates whether the
            classifier is a binary-classifier-only or not.

        multi_label : bool, default=False
            Whether the classifier supports multi-label output.
        """

        poor_score: bool = False
        multi_class: bool = True
        multi_label: bool = False

    @dataclass(**_dataclass_args())
    class RegressorTags:
        """Tags for the regressor.

        Parameters
        ----------
        poor_score : bool, default=False
            Whether the estimator fails to provide a "reasonable" test-set
            score, which currently for regression is an R2 of 0.5 on
            ``make_regression(n_samples=200, n_features=10,
            n_informative=1, bias=5.0, noise=20, random_state=42)``. The
            dataset and values are based on current estimators in scikit-learn
            and might be replaced by something more systematic.

        multi_label : bool, default=False
            Whether the regressor supports multilabel output.
        """

        poor_score: bool = False
        multi_label: bool = False

    @dataclass(**_dataclass_args())
    class Tags:
        """Tags for the estimator.

        See :ref:`estimator_tags` for more information.

        Parameters
        ----------
        estimator_type : str or None
            The type of the estimator. Can be one of:
            - "classifier"
            - "regressor"
            - "transformer"
            - "clusterer"
            - "outlier_detector"
            - "density_estimator"

        target_tags : :class:`TargetTags`
            The target(y) tags.

        transformer_tags : :class:`TransformerTags` or None
            The transformer tags.

        classifier_tags : :class:`ClassifierTags` or None
            The classifier tags.

        regressor_tags : :class:`RegressorTags` or None
            The regressor tags.

        array_api_support : bool, default=False
            Whether the estimator supports Array API compatible inputs.

        no_validation : bool, default=False
            Whether the estimator skips input-validation. This is only meant for
            stateless and dummy transformers!

        non_deterministic : bool, default=False
            Whether the estimator is not deterministic given a fixed ``random_state``.

        requires_fit : bool, default=True
            Whether the estimator requires to be fitted before calling one of
            `transform`, `predict`, `predict_proba`, or `decision_function`.

        _skip_test : bool, default=False
            Whether to skip common tests entirely. Don't use this unless
            you have a *very good* reason.

        input_tags : :class:`InputTags`
            The input data(X) tags.
        """

        estimator_type: str | None
        target_tags: TargetTags
        transformer_tags: TransformerTags | None = None
        classifier_tags: ClassifierTags | None = None
        regressor_tags: RegressorTags | None = None
        array_api_support: bool = False
        no_validation: bool = False
        non_deterministic: bool = False
        requires_fit: bool = True
        _skip_test: bool = False
        input_tags: InputTags = field(default_factory=InputTags)

    def _patched_more_tags(estimator, expected_failed_checks):
        import copy

        from sklearn.utils._tags import _safe_tags

        original_tags = copy.deepcopy(_safe_tags(estimator))

        def patched_more_tags(self):
            original_tags.update({"_xfail_checks": expected_failed_checks})
            return original_tags

        estimator.__class__._more_tags = patched_more_tags
        return estimator

    def check_estimator(
        estimator=None,
        generate_only=False,
        *,
        legacy: bool = True,
        expected_failed_checks: dict[str, str] | None = None,
        on_skip: Literal["warn"] | None = "warn",
        on_fail: Literal["raise", "warn"] | None = "raise",
        callback: Callable | None = None,
    ):
        # legacy, on_skip, on_fail, and callback are not supported and ignored
        from sklearn.utils.estimator_checks import check_estimator

        return check_estimator(
            _patched_more_tags(estimator, expected_failed_checks),
            generate_only=generate_only,
        )

    def parametrize_with_checks(
        estimators,
        *,
        legacy: bool = True,
        expected_failed_checks: Callable | None = None,
    ):
        # legacy is not supported and ignored
        from sklearn.utils.estimator_checks import parametrize_with_checks

        estimators = [
            _patched_more_tags(estimator, expected_failed_checks(estimator))
            for estimator in estimators
        ]

        return parametrize_with_checks(estimators)

else:
    # test_common
    # tags infrastructure
    from sklearn.utils import (
        ClassifierTags,
        InputTags,
        RegressorTags,
        Tags,
        TargetTags,
        TransformerTags,
    )
    from sklearn.utils._test_common.instance_generator import (
        _construct_instances,  # noqa: F401
    )
    from sklearn.utils.estimator_checks import (
        check_estimator,  # noqa: F401
        parametrize_with_checks,  # noqa: F401
    )
    from sklearn.utils.multiclass import type_of_target  # noqa: F401

    # validation
    from sklearn.utils.validation import (
        _check_feature_names,  # noqa: F401
        _check_n_features,  # noqa: F401
        validate_data,  # noqa: F401
    )
