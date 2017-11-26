# coding: utf-8
# pylint: skip-file
import os
import subprocess
import tempfile
import unittest

import lightgbm as lgb
import numpy as np
from sklearn.datasets import load_breast_cancer, dump_svmlight_file
from sklearn.model_selection import train_test_split


class TestBasic(unittest.TestCase):

    def test(self):
        X_train, X_test, y_train, y_test = train_test_split(*load_breast_cancer(True), test_size=0.1, random_state=2)
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = train_data.create_valid(X_test, label=y_test)

        params = {
            "objective": "binary",
            "metric": "auc",
            "min_data": 10,
            "num_leaves": 15,
            "verbose": -1,
            "num_threads": 1,
            "max_bin": 255
        }
        bst = lgb.Booster(params, train_data)
        bst.add_valid(valid_data, "valid_1")

        for i in range(30):
            bst.update()
            if i % 10 == 0:
                print(bst.eval_train(), bst.eval_valid())
        bst.save_model("model.txt")
        pred_from_matr = bst.predict(X_test)
        with tempfile.NamedTemporaryFile() as f:
            tname = f.name
        with open(tname, "w+b") as f:
            dump_svmlight_file(X_test, y_test, f)
        pred_from_file = bst.predict(tname)
        os.remove(tname)
        self.assertEqual(len(pred_from_matr), len(pred_from_file))
        for preds in zip(pred_from_matr, pred_from_file):
            self.assertAlmostEqual(*preds, places=15)

        # check saved model persistence
        bst = lgb.Booster(params, model_file="model.txt")
        pred_from_model_file = bst.predict(X_test)
        self.assertEqual(len(pred_from_matr), len(pred_from_model_file))
        for preds in zip(pred_from_matr, pred_from_model_file):
            # we need to check the consistency of model file here, so test for exact equal
            self.assertEqual(*preds)

        # check early stopping is working. Make it stop very early, so the scores should be very close to zero
        pred_parameter = {"pred_early_stop": True, "pred_early_stop_freq": 5, "pred_early_stop_margin": 1.5}
        pred_early_stopping = bst.predict(X_test, pred_parameter=pred_parameter)
        self.assertEqual(len(pred_from_matr), len(pred_early_stopping))
        for preds in zip(pred_early_stopping, pred_from_matr):
            # scores likely to be different, but prediction should still be the same
            self.assertEqual(preds[0] > 0, preds[1] > 0)
