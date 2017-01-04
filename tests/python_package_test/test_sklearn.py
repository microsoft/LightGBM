# coding: utf-8
# pylint: skip-file
import unittest

import lightgbm as lgb
import numpy as np
from sklearn.base import clone
from sklearn.datasets import (load_boston, load_breast_cancer, load_digits,
                              load_svmlight_file)
from sklearn.externals import joblib
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split


def test_template(X_y=load_boston(True), model=lgb.LGBMRegressor,
                  feval=mean_squared_error, num_round=100,
                  custom_obj=None, predict_proba=False,
                  return_data=False, return_model=False):
    X_train, X_test, y_train, y_test = train_test_split(*X_y, test_size=0.1, random_state=42)
    if return_data:
        return X_train, X_test, y_train, y_test
    arguments = {'n_estimators': num_round, 'silent': True}
    if custom_obj:
        arguments['objective'] = custom_obj
    gbm = model(**arguments)
    gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)
    if return_model:
        return gbm
    elif predict_proba:
        return feval(y_test, gbm.predict_proba(X_test))
    else:
        return feval(y_test, gbm.predict(X_test))


class TestSklearn(unittest.TestCase):

    def test_binary(self):
        X_y = load_breast_cancer(True)
        ret = test_template(X_y, lgb.LGBMClassifier, log_loss, predict_proba=True)
        self.assertLess(ret, 0.15)

    def test_regreesion(self):
        self.assertLess(test_template() ** 0.5, 4)

    def test_multiclass(self):
        X_y = load_digits(10, True)

        def multi_error(y_true, y_pred):
            return np.mean(y_true != y_pred)
        ret = test_template(X_y, lgb.LGBMClassifier, multi_error)
        self.assertLess(ret, 0.2)

    def test_lambdarank(self):
        X_train, y_train = load_svmlight_file('../../examples/lambdarank/rank.train')
        X_test, y_test = load_svmlight_file('../../examples/lambdarank/rank.test')
        q_train = np.loadtxt('../../examples/lambdarank/rank.train.query')
        q_test = np.loadtxt('../../examples/lambdarank/rank.test.query')
        lgb_model = lgb.LGBMRanker().fit(X_train, y_train,
                                         group=q_train,
                                         eval_set=[(X_test, y_test)],
                                         eval_group=[q_test],
                                         eval_at=[1],
                                         verbose=False,
                                         callbacks=[lgb.reset_parameter(learning_rate=lambda x: 0.95 ** x * 0.1)])

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

    def test_dart(self):
        X_train, X_test, y_train, y_test = test_template(return_data=True)
        gbm = lgb.LGBMRegressor(boosting_type='dart')
        gbm.fit(X_train, y_train)
        self.assertLessEqual(gbm.score(X_train, y_train), 1.)

    def test_grid_search(self):
        X_train, X_test, y_train, y_test = test_template(return_data=True)
        params = {'boosting_type': ['dart', 'gbdt'],
                  'n_estimators': [15, 20],
                  'drop_rate': [0.1, 0.2]}
        gbm = GridSearchCV(lgb.LGBMRegressor(), params, cv=3)
        gbm.fit(X_train, y_train)
        self.assertIn(gbm.best_params_['n_estimators'], [15, 20])

    def test_clone_and_property(self):
        gbm = test_template(return_model=True)
        gbm_clone = clone(gbm)
        self.assertIsInstance(gbm.booster_, lgb.Booster)
        self.assertIsInstance(gbm.feature_importance_, np.ndarray)
        clf = test_template(load_digits(2, True), model=lgb.LGBMClassifier, return_model=True)
        self.assertListEqual(sorted(clf.classes_), [0, 1])
        self.assertEqual(clf.n_classes_, 2)
        self.assertIsInstance(clf.booster_, lgb.Booster)
        self.assertIsInstance(clf.feature_importance_, np.ndarray)

    def test_joblib(self):
        gbm = test_template(num_round=10, return_model=True)
        joblib.dump(gbm, 'lgb.pkl')
        gbm_pickle = joblib.load('lgb.pkl')
        self.assertIsInstance(gbm_pickle.booster_, lgb.Booster)
        self.assertDictEqual(gbm.get_params(), gbm_pickle.get_params())
        self.assertListEqual(list(gbm.feature_importance_), list(gbm_pickle.feature_importance_))
        X_train, X_test, y_train, y_test = test_template(return_data=True)
        gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        gbm_pickle.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        self.assertDictEqual(gbm.evals_result_, gbm_pickle.evals_result_)
        pred_origin = gbm.predict(X_test)
        pred_pickle = gbm_pickle.predict(X_test)
        self.assertEqual(len(pred_origin), len(pred_pickle))
        for preds in zip(pred_origin, pred_pickle):
            self.assertAlmostEqual(*preds, places=5)


print("----------------------------------------------------------------------")
print("running test_sklearn.py")
unittest.main()
