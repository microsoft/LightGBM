# coding: utf-8
from zoneinfo import ZoneInfo

import pytest

import lightgbm as lgb

pd = pytest.importorskip("pandas")


@pytest.mark.parametrize(
    ("dtype", "values", "init_args", "valid"),
    [
        # Valid dtypes
        (pd.Int8Dtype, [1, 2, 3], {}, True),
        (pd.Int16Dtype, [1, 2, 3], {}, True),
        (pd.Int32Dtype, [1, 2, 3], {}, True),
        (pd.Int64Dtype, [1, 2, 3], {}, True),
        (pd.UInt8Dtype, [1, 2, 3], {}, True),
        (pd.UInt16Dtype, [1, 2, 3], {}, True),
        (pd.UInt32Dtype, [1, 2, 3], {}, True),
        (pd.UInt64Dtype, [1, 2, 3], {}, True),
        (pd.Float32Dtype, [1.0, 2.0, 3.0], {}, True),
        (pd.Float64Dtype, [1.0, 2.0, 3.0], {}, True),
        (pd.BooleanDtype, [True, False, True], {}, True),
        (pd.SparseDtype, [1.0, 2.0, 3.0], {}, True),
        (pd.CategoricalDtype, ["a", "b", "c"], {"ordered": False}, True),
        (pd.CategoricalDtype, ["x", "y", "z"], {"ordered": True}, True),
        # Invalid dtypes
        (pd.StringDtype, ["a", "b", "c"], {}, False),
        (pd.DatetimeTZDtype, ["2020-01-01", "2020-01-02", "2020-01-03"], {"tz": ZoneInfo("UTC")}, False),
        (pd.PeriodDtype, [pd.Period("2024"), pd.Period("2025"), pd.Period("2026")], {"freq": "Y"}, False),
        (pd.IntervalDtype, [pd.Interval(0, 1), pd.Interval(1, 2), pd.Interval(2, 3)], {"subtype": "int64"}, False),
    ],
)
def test_narwhals_dtype_validation_for_pandas(dtype, values, init_args, valid):
    """Valid dtypes should construct; invalid dtypes should raise ValueError."""
    df = pd.DataFrame({"col": pd.Series(values, dtype=dtype(**init_args)), "num_col": [1.0, 2.0, 3.0]})
    y = [0, 1, 0]

    if valid:
        lgb.Dataset(df, label=y).construct()
    else:
        with pytest.raises(ValueError, match="DataFrame dtypes must be int, float, bool, categorical or enum"):
            lgb.Dataset(df, label=y).construct()
