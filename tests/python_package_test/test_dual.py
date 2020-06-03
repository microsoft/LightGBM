"""Tests for dual GPU+CPU support."""

import os
import pytest

import lightgbm as lgb
import numpy as np

@pytest.mark.skipif(os.environ.get("LIGHTGBM_TEST_DUAL_CPU_GPU", None) is None,
                    reason="Only run if appropriate env variable is set")
@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_both_gpu_and_cpu(device):
    """If compiled appropriately, the same installation will support both GPU and CPU."""
    data = np.random.rand(500, 10)
    label = np.random.randint(2, size=500)
    validation_data = train_data = lgb.Dataset(data, label=label)

    param = {'num_leaves': 31, 'objective': 'binary', 'device': device}
    # This will raise an exception if it's an unsupported device:
    lgb.train(param, train_data, 10, valid_sets=[validation_data])