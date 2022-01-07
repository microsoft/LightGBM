# coding: utf-8
import filecmp
import numbers
from pathlib import Path

import numpy as np
# import pytest
from scipy import sparse
from sklearn.datasets import dump_svmlight_file, load_svmlight_file, load_breast_cancer
from sklearn.model_selection import train_test_split
import sys
sys.path.append("python-package")
import lightgbm as lgb
from lightgbm.compat import PANDAS_INSTALLED, pd_DataFrame, pd_Series
DATA_DIR = Path(__file__).absolute().parents[1] / 'data'
# from utils import load_breast_cancer
# import scipy.sparse
import pandas as pd

use_num = False

def test_string_basic(tmp_path):

    if use_num:
        X_train = pd.read_csv(DATA_DIR / 'train_drug_num.csv')
        X_test = pd.read_csv(DATA_DIR / 'test_drug_num.csv')
    else:
        X_train = pd.read_csv(DATA_DIR / 'train_drug.csv')
        X_test = pd.read_csv(DATA_DIR / 'test_drug.csv')
    
    y_train = X_train.pop('Drug')
    y_test = X_test.pop('Drug')

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = train_data.create_valid(X_test, label=y_test)

    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "metric_freq": 1,
        "is_training_metric": True,
        "min_data": 10,
        "num_leaves": 63,
        "verbose": -1,
        "num_threads": 1,
        "max_bin": 255,
        "learning_rate": 0.1,
        "gpu_use_dp": True,
        "feature_fraction": 0.8,
        "categorical_feature": [1, 2, 3],
        "bagging_freq": 5,
        "bagging_fraction": 0.8,
        "min_data_in_leaf": 50,
        "min_sum_hessian_in_leaf": 5.0,
        "is_enable_sparse": True,
        "use_two_round_loading": False,
        "is_save_binary_file": False,
        "tree_learner": "serial",
        "num_trees": 100,
        "boosting_type": "gbdt",
        
    }
    bst = lgb.Booster(params, train_data)
    bst.add_valid(valid_data, "valid_1")

    for i in range(20):
        bst.update()
        if i % 10 == 0:
            print(bst.eval_train(), bst.eval_valid())

    feature_names = list(X_train.columns)
    # feature_names = [f"Column_{i}" for i in range(X_train.shape[1])]
    assert train_data.get_feature_name() == feature_names

    if use_num:
        model_file = tmp_path / "model_num.txt"
    else:
        model_file = tmp_path / "model.txt"

    bst.save_model(model_file)
    pred_from_matr = bst.predict(X_test)

    # check saved model persistence
    bst = lgb.Booster(params, model_file=model_file)
    assert bst.feature_name() == feature_names
    pred_from_model_file = bst.predict(X_test)
    # we need to check the consistency of model file here, so test for exact equal
    np.testing.assert_array_equal(pred_from_matr, pred_from_model_file)

    # check early stopping is working. Make it stop very early, so the scores should be very close to zero
    pred_parameter = {
        "pred_early_stop": True,
        "pred_early_stop_freq": 5,
        "pred_early_stop_margin": 1.5
    }
    pred_early_stopping = bst.predict(X_test, **pred_parameter)
    # scores likely to be different, but prediction should still be the same
    np.testing.assert_array_equal(np.sign(pred_from_matr),
                                  np.sign(pred_early_stopping))


# test_basic(Path(r'tests\python_package_test\test_string'))