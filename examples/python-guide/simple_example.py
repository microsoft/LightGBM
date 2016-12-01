import numpy as np
import random
import lightgbm as lgb
from sklearn import datasets, metrics, model_selection

rng = np.random.RandomState(2016)

X, y = datasets.make_classification(n_samples=10000, n_features=100)
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=1)
lgb_model = lgb.LGBMClassifier(n_estimators=100).fit(x_train, y_train, [(x_test, y_test)], eval_metric="auc")
