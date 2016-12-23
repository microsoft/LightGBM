# coding: utf-8
# pylint: skip-file
import os, unittest
import numpy as np
import lightgbm as lgb
from sklearn.metrics import log_loss, mean_squared_error, mean_absolute_error
from sklearn.datasets import load_breast_cancer, load_boston, load_digits, load_iris, load_svmlight_file
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import clone

def test_template(X_y=load_boston(True), model=lgb.LGBMRegressor,
                feval=mean_squared_error, stratify=None, num_round=100, return_data=False,
                return_model=False, init_model=None, custom_obj=None, proba=False):
    X, y = X_y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                        stratify=stratify,
                                                        random_state=42)
    if return_data: return X_train, X_test, y_train, y_test
    gbm = model(n_estimators=num_round, objective=custom_obj) if custom_obj else model(n_estimators=num_round)
    gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)
    if return_model: return gbm
    else: return feval(y_test, gbm.predict_proba(X_test) if proba else gbm.predict(X_test))

class TestSklearn(unittest.TestCase):

    def test_binary(self):
        X_y= load_breast_cancer(True)
        ret = test_template(X_y, lgb.LGBMClassifier, log_loss, stratify=X_y[1], proba=True)
        self.assertLess(ret, 0.15)

    def test_regreesion(self):
        self.assertLess(test_template() ** 0.5, 4)
 
    def test_multiclass(self):
        X_y = load_digits(10, True)
        def multi_error(y_true, y_pred):
            return np.mean(y_true != y_pred)
        ret = test_template(X_y, lgb.LGBMClassifier, multi_error, stratify=X_y[1])
        self.assertLess(ret, 0.2)
        
    def test_lambdarank(self):
        X_train, y_train = load_svmlight_file('../../examples/lambdarank/rank.train')
        X_test, y_test = load_svmlight_file('../../examples/lambdarank/rank.test')
        q_train = np.loadtxt('../../examples/lambdarank/rank.train.query')
        lgb_model = lgb.LGBMRanker().fit(X_train, y_train, group=q_train, eval_at=[1])

    def test_regression_with_custom_objective(self):
        def objective_ls(y_true, y_pred):
            grad = (y_pred - y_true)
            hess = np.ones(len(y_true))
            return grad, hess
        ret = test_template(custom_obj=objective_ls)
        self.assertLess(ret, 100)

    def test_binary_classification_with_custom_objective(self):
        def logregobj(y_true, y_pred):
            y_pred = 1.0 / (1.0 + np.exp(-y_pred))
            grad = y_pred - y_true
            hess = y_pred * (1.0 - y_pred)
            return grad, hess
        X_y = load_digits(2, True)
        def binary_error(y_test, y_pred):
            return np.mean([int(p > 0.5) != y for y, p in zip(y_test, y_pred)])
        ret = test_template(X_y, lgb.LGBMClassifier, feval=binary_error, custom_obj=logregobj)
        self.assertLess(ret, 0.1)

    def test_grid_search(self):
        X_train, X_test, y_train, y_test = test_template(return_data=True)
        params = {'n_estimators': [10, 15, 20]}
        gbm = GridSearchCV(lgb.LGBMRegressor(), params, cv=5)
        gbm.fit(X_train, y_train)
        self.assertIn(gbm.best_params_['n_estimators'], [10, 15, 20])

    def test_clone(self):
        gbm = test_template(return_model=True)
        gbm_clone = clone(gbm)

print("----------------------------------------------------------------------")
print("running test_sklearn.py")
unittest.main()
