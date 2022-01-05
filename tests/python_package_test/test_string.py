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

# from utils import load_breast_cancer
# import scipy.sparse
import pandas as pd

def test_basic(tmp_path):
    # X_train, X_test, y_train, y_test = train_test_split(
    #     *load_breast_cancer(return_X_y=True), test_size=0.1, random_state=2)

    # X_train_copy = np.array([np.array(['x' for i in range(15)] + ['y' for i in range(15)], dtype=object) for i in range (512)])
    # # X_test_copy = np.array([np.array(['x' for i in range(15)] + ['y' for i in range(15)], dtype=object) for i in range (57)])
    # y_train_copy = np.array(['x' for i in range(256)] + ['y' for i in range(256)])
    # feature_names = [f"Column_{i}" for i in range(X_train.shape[1])]
    import os
    print(os.getcwd())
    X_train = pd.read_csv(r'tests\data\train.csv')
    y_train = X_train.pop('buys')
    X_test = pd.read_csv(r'tests\data\test.csv')
    y_test = X_test.pop('buys')

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = train_data.create_valid(X_test, label=y_test)

    # train_data = lgb.Dataset(r'D:\Documents\MSRA\OneDrive - Microsoft\LightGBM\train_normal.csv')
    # valid_data = train_data.create_valid(
    #     r'D:\Documents\MSRA\OneDrive - Microsoft\LightGBM\test_normal.csv')

    params = {
        "objective": "binary",
        "metric": "auc",
        "min_data": 10,
        "num_leaves": 15,
        "verbose": -1,
        "num_threads": 1,
        "max_bin": 255,
        "gpu_use_dp": True
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

    # assert bst.current_iteration() == 20
    # assert bst.num_trees() == 20
    # assert bst.num_model_per_iteration() == 1
    # assert bst.lower_bound() == pytest.approx(-2.9040190126976606)
    # assert bst.upper_bound() == pytest.approx(3.3182142872462883)

    tname = tmp_path / "svm_light.dat"
    # mapping_file = tmp_path / "mapping.json"
    model_file = tmp_path / "model.txt"

    bst.save_model(model_file)
    pred_from_matr = bst.predict(X_test)



    # with open(tname, "w+b") as f:
    #     dump_svmlight_file(X_test, y_test, f)

    # pred_from_file = bst.predict(tname)
    # np.testing.assert_allclose(pred_from_matr, pred_from_file)

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

    # test that shape is checked during prediction
    # bad_X_test = X_test[:, 1:]
    # bad_shape_error_msg = "The number of features in data*"
    # np.testing.assert_raises_regex(lgb.basic.LightGBMError,
    #                                bad_shape_error_msg, bst.predict,
    #                                bad_X_test)
    # np.testing.assert_raises_regex(lgb.basic.LightGBMError,
    #                                bad_shape_error_msg, bst.predict,
    #                                sparse.csr_matrix(bad_X_test))
    # np.testing.assert_raises_regex(lgb.basic.LightGBMError,
    #                                bad_shape_error_msg, bst.predict,
    #                                sparse.csc_matrix(bad_X_test))
    # with open(tname, "w+b") as f:
    #     dump_svmlight_file(bad_X_test, y_test, f)
    # np.testing.assert_raises_regex(lgb.basic.LightGBMError,
    #                                bad_shape_error_msg, bst.predict, tname)
    # with open(tname, "w+b") as f:
    #     dump_svmlight_file(X_test, y_test, f, zero_based=False)
    # np.testing.assert_raises_regex(lgb.basic.LightGBMError,
    #                                bad_shape_error_msg, bst.predict, tname)


test_basic(Path('test_string'))