# coding: utf-8
import numpy as np
from sklearn import datasets, metrics, model_selection
import lightgbm as lgb

X, Y = datasets.make_classification(n_samples=100000, n_features=100)
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.1)

train_data = lgb.Dataset(x_train, max_bin=255, label=y_train)

valid_data = train_data.create_valid(x_test, label=y_test)

config={"objective":"binary","metric":"auc", "min_data":1, "num_leaves":15}
bst = lgb.Booster(params=config, train_set=train_data)
bst.add_valid(valid_data,"valid_1")

for i in range(100):
	bst.update()
	if i % 10 == 0:
		print(bst.eval_train())
		print(bst.eval_valid())
bst.save_model("model.txt")

