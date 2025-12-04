# adopted from tests/test_dual.py

import os
import platform

import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import log_loss

import lightgbm as lgb

X, y = load_breast_cancer(return_X_y=True)
data = lgb.Dataset(X, y, params={"verbosity": 10})

params_cpu = {"verbosity": 10, "num_leaves": 31, "objective": "binary", "device": "cpu"}
cpu_bst = lgb.train(params_cpu, data, num_boost_round=10)
cpu_score = log_loss(y, cpu_bst.predict(X))

params_gpu = params_cpu.copy()
params_gpu["device"] = "gpu"
# Double-precision floats are only supported on x86_64 with PoCL
params_gpu["gpu_use_dp"] = platform.machine() == "x86_64"
gpu_bst = lgb.train(params_gpu, data, num_boost_round=10)
gpu_score = log_loss(y, gpu_bst.predict(X))

rel = 1e-6 if params_gpu["gpu_use_dp"] else 1e-4
assert cpu_score == pytest.approx(gpu_score, rel=rel)
assert gpu_score < 0.242
