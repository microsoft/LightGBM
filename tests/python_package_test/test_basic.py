# coding: utf-8
# pylint: skip-file
import os
import tempfile
import unittest

import lightgbm as lgb
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


class TestBasic(unittest.TestCase):

    def test(self):
        X_train, X_test, y_train, y_test = train_test_split(*load_breast_cancer(True), test_size=0.1, random_state=1)
        train_data = lgb.Dataset(X_train, max_bin=255, label=y_train)
        valid_data = train_data.create_valid(X_test, label=y_test)

        params = {
            "objective": "binary",
            "metric": "auc",
            "min_data": 1,
            "num_leaves": 15,
            "verbose": -1
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
            np.savetxt(f, X_test, delimiter=',')
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
            self.assertEqual(*preds)


print("----------------------------------------------------------------------")
print("running test_basic.py")
unittest.main()
