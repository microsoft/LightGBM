# coding: utf-8
# pylint: skip-file
import os
import unittest

import lightgbm as lgb
import numpy as np
from sklearn.datasets import load_svmlight_file


class FileLoader(object):

    def __init__(self, directory, prefix, config_file='train.conf'):
        directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), directory)
        self.directory = directory
        self.prefix = prefix
        self.params = {'gpu_use_dp': True}
        with open(os.path.join(directory, config_file), 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = [token.strip() for token in line.split('=')]
                    if 'early_stopping' not in key:  # disable early_stopping
                        self.params[key] = value if key != 'num_trees' else int(value)

    def load_dataset(self, suffix, is_sparse=False):
        filename = self.path(suffix)
        if is_sparse:
            X, Y = load_svmlight_file(filename, dtype=np.float64, zero_based=True)
            return X, Y, filename
        else:
            mat = np.loadtxt(filename, dtype=np.float64)
            return mat[:, 1:], mat[:, 0], filename

    def load_field(self, suffix):
        return np.loadtxt(os.path.join(self.directory, self.prefix + suffix))

    def load_cpp_result(self, result_file='LightGBM_predict_result.txt'):
        return np.loadtxt(os.path.join(self.directory, result_file))

    def train_predict_check(self, lgb_train, X_test, X_test_fn, sk_pred):
        gbm = lgb.train(self.params, lgb_train)
        y_pred = gbm.predict(X_test)
        cpp_pred = gbm.predict(X_test_fn)
        np.testing.assert_allclose(y_pred, cpp_pred)
        np.testing.assert_allclose(y_pred, sk_pred)

    def file_load_check(self, lgb_train, name):
        lgb_train_f = lgb.Dataset(self.path(name), params=self.params).construct()
        for f in ('num_data', 'num_feature', 'get_label', 'get_weight', 'get_init_score', 'get_group'):
            a = getattr(lgb_train, f)()
            b = getattr(lgb_train_f, f)()
            if a is None and b is None:
                pass
            elif a is None:
                assert np.all(b == 1), f
            elif isinstance(b, (list, np.ndarray)):
                np.testing.assert_allclose(a, b)
            else:
                assert a == b, f

    def path(self, suffix):
        return os.path.join(self.directory, self.prefix + suffix)


class TestEngine(unittest.TestCase):

    def test_binary(self):
        fd = FileLoader('../../examples/binary_classification', 'binary')
        X_train, y_train, _ = fd.load_dataset('.train')
        X_test, _, X_test_fn = fd.load_dataset('.test')
        weight_train = fd.load_field('.train.weight')
        lgb_train = lgb.Dataset(X_train, y_train, params=fd.params, weight=weight_train)
        gbm = lgb.LGBMClassifier(**fd.params)
        gbm.fit(X_train, y_train, sample_weight=weight_train)
        sk_pred = gbm.predict_proba(X_test)[:, 1]
        fd.train_predict_check(lgb_train, X_test, X_test_fn, sk_pred)
        fd.file_load_check(lgb_train, '.train')

    def test_multiclass(self):
        fd = FileLoader('../../examples/multiclass_classification', 'multiclass')
        X_train, y_train, _ = fd.load_dataset('.train')
        X_test, _, X_test_fn = fd.load_dataset('.test')
        lgb_train = lgb.Dataset(X_train, y_train)
        gbm = lgb.LGBMClassifier(**fd.params)
        gbm.fit(X_train, y_train)
        sk_pred = gbm.predict_proba(X_test)
        fd.train_predict_check(lgb_train, X_test, X_test_fn, sk_pred)
        fd.file_load_check(lgb_train, '.train')

    def test_regression(self):
        fd = FileLoader('../../examples/regression', 'regression')
        X_train, y_train, _ = fd.load_dataset('.train')
        X_test, _, X_test_fn = fd.load_dataset('.test')
        init_score_train = fd.load_field('.train.init')
        lgb_train = lgb.Dataset(X_train, y_train, init_score=init_score_train)
        gbm = lgb.LGBMRegressor(**fd.params)
        gbm.fit(X_train, y_train, init_score=init_score_train)
        sk_pred = gbm.predict(X_test)
        fd.train_predict_check(lgb_train, X_test, X_test_fn, sk_pred)
        fd.file_load_check(lgb_train, '.train')

    def test_lambdarank(self):
        fd = FileLoader('../../examples/lambdarank', 'rank')
        X_train, y_train, _ = fd.load_dataset('.train', is_sparse=True)
        X_test, _, X_test_fn = fd.load_dataset('.test', is_sparse=True)
        group_train = fd.load_field('.train.query')
        lgb_train = lgb.Dataset(X_train, y_train, group=group_train)
        gbm = lgb.LGBMRanker(**fd.params)
        gbm.fit(X_train, y_train, group=group_train)
        sk_pred = gbm.predict(X_test)
        fd.train_predict_check(lgb_train, X_test, X_test_fn, sk_pred)
        fd.file_load_check(lgb_train, '.train')
