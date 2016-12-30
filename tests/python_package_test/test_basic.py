# coding: utf-8
# pylint: skip-file
import unittest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import lightgbm as lgb

class TestBasic(unittest.TestCase):

    def test(self):
        X_train, X_test, y_train, y_test = train_test_split(*load_breast_cancer(True), test_size=0.1)

        train_data = lgb.Dataset(X_train, max_bin=255, label=y_train, free_raw_data=False)
        valid_data = train_data.create_valid(X_test, label=y_test)
		# train_data.save_binary('train.bin')

        params = {
            "objective" : "binary",
            "metric" : "auc",
            "min_data" : 1,
            "num_leaves" : 15,
            "verbose" : -1
        }
        bst = lgb.Booster(params, train_data)
        bst.add_valid(valid_data, "valid_1")

        for i in range(30):
            bst.update()
            if i % 10 == 0:
                print(bst.eval_train(), bst.eval_valid())
        bst.save_model("model.txt")

print("----------------------------------------------------------------------")
print("running test_basic.py")
unittest.main()
