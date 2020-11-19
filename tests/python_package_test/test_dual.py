"""Tests for dual GPU+CPU support."""

import os
import pytest

import lightgbm as lgb
import numpy as np
from lightgbm.basic import LightGBMError


@pytest.mark.skipif(
    os.environ.get("LIGHTGBM_TEST_DUAL_CPU_GPU", None) is None,
    reason="Only run if appropriate env variable is set",
)
def test_cpu_works():
    """If compiled appropriately, the same installation will support both GPU and CPU."""
    data = np.random.rand(500, 10)
    label = np.random.randint(2, size=500)
    validation_data = train_data = lgb.Dataset(data, label=label)
    param = {"num_leaves": 31, "objective": "binary", "device": "cpu"}

    # With the properly built library, running on the cpu should work
    # whether an OpenCL runtime is installed (LIGHTGBM_TEST_DUAL_CPU_GPU="2")
    # or not (LIGHTGBM_TEST_DUAL_CPU_GPU="1")
    gbm = lgb.train(param, train_data, 10, valid_sets=[validation_data])


@pytest.mark.skipif(
    os.environ.get("LIGHTGBM_TEST_DUAL_CPU_GPU", None) is None,
    reason="Only run if appropriate env variable is set",
)
def test_gpu_works():
    """If compiled appropriately, the same installation will support both GPU and CPU."""
    TEST_DUAL_MODE = os.getenv('LIGHTGBM_TEST_DUAL_CPU_GPU')

    data = np.random.rand(500, 10)
    label = np.random.randint(2, size=500)
    validation_data = train_data = lgb.Dataset(data, label=label)
    param = {"num_leaves": 31, "objective": "binary", "device": "gpu"}

    if TEST_DUAL_MODE == "1":
        # We do NOT expect OpenCL to be installed,
        # so we expect train({'device': 'gpu'}) to fail gracefully ...
        try:
            gbm = lgb.train(param, train_data, 10, valid_sets=[validation_data])
        except LightGBMError as e:
            if str(e) == "No OpenCL device found": # ... with this message
                pass
            else:
                # We do not expect it to fail with a different message
                raise
        else:
            # We do not expect it to succeed
            raise
    else:
        # MODE must be "2", so we expect OpenCL to be installed and
        # train({'device': 'gpu'}) to run successfully
        gbm = lgb.train(param, train_data, 10, valid_sets=[validation_data])
