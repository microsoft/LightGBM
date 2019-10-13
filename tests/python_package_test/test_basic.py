# coding: utf-8
# pylint: skip-file
import os
import tempfile
import unittest

import lightgbm as lgb
import numpy as np

from scipy import sparse
from sklearn.datasets import load_breast_cancer, dump_svmlight_file, load_svmlight_file
from sklearn.model_selection import train_test_split


class TestBasic(unittest.TestCase):

    def test(self):
        X_train, X_test, y_train, y_test = train_test_split(*load_breast_cancer(True),
                                                            test_size=0.1, random_state=2)
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

        self.assertEqual(bst.current_iteration(), 30)
        self.assertEqual(bst.num_trees(), 30)
        self.assertEqual(bst.num_model_per_iteration(), 1)

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
        os.remove("model.txt")
        pred_from_model_file = bst.predict(X_test)
        self.assertEqual(len(pred_from_matr), len(pred_from_model_file))
        for preds in zip(pred_from_matr, pred_from_model_file):
            # we need to check the consistency of model file here, so test for exact equal
            self.assertEqual(*preds)

        # check early stopping is working. Make it stop very early, so the scores should be very close to zero
        pred_parameter = {"pred_early_stop": True, "pred_early_stop_freq": 5, "pred_early_stop_margin": 1.5}
        pred_early_stopping = bst.predict(X_test, **pred_parameter)
        self.assertEqual(len(pred_from_matr), len(pred_early_stopping))
        for preds in zip(pred_early_stopping, pred_from_matr):
            # scores likely to be different, but prediction should still be the same
            self.assertEqual(preds[0] > 0, preds[1] > 0)

        # test that shape is checked during prediction
        bad_X_test = X_test[:, 1:]
        bad_shape_error_msg = "The number of features in data*"
        np.testing.assert_raises_regex(lgb.basic.LightGBMError, bad_shape_error_msg,
                                       bst.predict, bad_X_test)
        np.testing.assert_raises_regex(lgb.basic.LightGBMError, bad_shape_error_msg,
                                       bst.predict, sparse.csr_matrix(bad_X_test))
        np.testing.assert_raises_regex(lgb.basic.LightGBMError, bad_shape_error_msg,
                                       bst.predict, sparse.csc_matrix(bad_X_test))
        with open(tname, "w+b") as f:
            dump_svmlight_file(bad_X_test, y_test, f)
        np.testing.assert_raises_regex(lgb.basic.LightGBMError, bad_shape_error_msg,
                                       bst.predict, tname)
        with open(tname, "w+b") as f:
            dump_svmlight_file(X_test, y_test, f, zero_based=False)
        np.testing.assert_raises_regex(lgb.basic.LightGBMError, bad_shape_error_msg,
                                       bst.predict, tname)
        os.remove(tname)

    def test_chunked_dataset(self):
        X_train, X_test, y_train, y_test = train_test_split(*load_breast_cancer(True), test_size=0.1, random_state=2)

        chunk_size = X_train.shape[0] // 10 + 1
        X_train = [X_train[i * chunk_size:(i + 1) * chunk_size, :] for i in range(X_train.shape[0] // chunk_size + 1)]
        X_test = [X_test[i * chunk_size:(i + 1) * chunk_size, :] for i in range(X_test.shape[0] // chunk_size + 1)]

        train_data = lgb.Dataset(X_train, label=y_train, params={"bin_construct_sample_cnt": 100})
        valid_data = train_data.create_valid(X_test, label=y_test, params={"bin_construct_sample_cnt": 100})

        train_data.construct()
        valid_data.construct()

    def test_subset_group(self):
        X_train, y_train = load_svmlight_file(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                           '../../examples/lambdarank/rank.train'))
        q_train = np.loadtxt(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                          '../../examples/lambdarank/rank.train.query'))
        lgb_train = lgb.Dataset(X_train, y_train, group=q_train)
        self.assertEqual(len(lgb_train.get_group()), 201)
        subset = lgb_train.subset(list(lgb.compat.range_(10))).construct()
        subset_group = subset.get_group()
        self.assertEqual(len(subset_group), 2)
        self.assertEqual(subset_group[0], 1)
        self.assertEqual(subset_group[1], 9)

    def test_add_features_throws_if_num_data_unequal(self):
        X1 = np.random.random((1000, 1))
        X2 = np.random.random((100, 1))
        d1 = lgb.Dataset(X1).construct()
        d2 = lgb.Dataset(X2).construct()
        with self.assertRaises(lgb.basic.LightGBMError):
            d1.add_features_from(d2)

    def test_add_features_throws_if_datasets_unconstructed(self):
        X1 = np.random.random((1000, 1))
        X2 = np.random.random((1000, 1))
        with self.assertRaises(ValueError):
            d1 = lgb.Dataset(X1)
            d2 = lgb.Dataset(X2)
            d1.add_features_from(d2)
        with self.assertRaises(ValueError):
            d1 = lgb.Dataset(X1).construct()
            d2 = lgb.Dataset(X2)
            d1.add_features_from(d2)
        with self.assertRaises(ValueError):
            d1 = lgb.Dataset(X1)
            d2 = lgb.Dataset(X2).construct()
            d1.add_features_from(d2)

    def test_add_features_equal_data_on_alternating_used_unused(self):
        X = np.random.random((1000, 5))
        X[:, [1, 3]] = 0
        names = ['col_%d' % i for i in range(5)]
        for j in range(1, 5):
            d1 = lgb.Dataset(X[:, :j], feature_name=names[:j]).construct()
            d2 = lgb.Dataset(X[:, j:], feature_name=names[j:]).construct()
            d1.add_features_from(d2)
            with tempfile.NamedTemporaryFile() as f:
                d1name = f.name
            d1._dump_text(d1name)
            d = lgb.Dataset(X, feature_name=names).construct()
            with tempfile.NamedTemporaryFile() as f:
                dname = f.name
            d._dump_text(dname)
            with open(d1name, 'rt') as d1f:
                d1txt = d1f.read()
            with open(dname, 'rt') as df:
                dtxt = df.read()
            os.remove(dname)
            os.remove(d1name)
            self.assertEqual(dtxt, d1txt)

    def test_add_features_same_booster_behaviour(self):
        X = np.random.random((1000, 5))
        X[:, [1, 3]] = 0
        names = ['col_%d' % i for i in range(5)]
        for j in range(1, 5):
            d1 = lgb.Dataset(X[:, :j], feature_name=names[:j]).construct()
            d2 = lgb.Dataset(X[:, j:], feature_name=names[j:]).construct()
            d1.add_features_from(d2)
            d = lgb.Dataset(X, feature_name=names).construct()
            y = np.random.random(1000)
            d1.set_label(y)
            d.set_label(y)
            b1 = lgb.Booster(train_set=d1)
            b = lgb.Booster(train_set=d)
            for k in range(10):
                b.update()
                b1.update()
            with tempfile.NamedTemporaryFile() as df:
                dname = df.name
            with tempfile.NamedTemporaryFile() as d1f:
                d1name = d1f.name
            b1.save_model(d1name)
            b.save_model(dname)
            with open(dname, 'rt') as df:
                dtxt = df.read()
            with open(d1name, 'rt') as d1f:
                d1txt = d1f.read()
            self.assertEqual(dtxt, d1txt)

    def test_get_feature_penalty_and_monotone_constraints(self):
        X = np.random.random((1000, 1))
        d = lgb.Dataset(X, params={'feature_penalty': [0.5],
                                   'monotone_constraints': [1]}).construct()
        np.testing.assert_allclose(d.get_feature_penalty(), [0.5])
        np.testing.assert_array_equal(d.get_monotone_constraints(), [1])
        d = lgb.Dataset(X).construct()
        self.assertIsNone(d.get_feature_penalty())
        self.assertIsNone(d.get_monotone_constraints())

    def test_add_features_feature_penalty(self):
        X = np.random.random((1000, 2))
        test_cases = [
            (None, None, None),
            ([0.5], None, [0.5, 1]),
            (None, [0.5], [1, 0.5]),
            ([0.5], [0.5], [0.5, 0.5])]
        for (p1, p2, expected) in test_cases:
            params1 = {'feature_penalty': p1} if p1 is not None else {}
            d1 = lgb.Dataset(X[:, 0].reshape((-1, 1)), params=params1).construct()
            params2 = {'feature_penalty': p2} if p2 is not None else {}
            d2 = lgb.Dataset(X[:, 1].reshape((-1, 1)), params=params2).construct()
            d1.add_features_from(d2)
            actual = d1.get_feature_penalty()
            if expected is None:
                self.assertIsNone(actual)
            else:
                np.testing.assert_allclose(actual, expected)

    def test_add_features_monotone_types(self):
        X = np.random.random((1000, 2))
        test_cases = [
            (None, None, None),
            ([1], None, [1, 0]),
            (None, [1], [0, 1]),
            ([1], [-1], [1, -1])]
        for (p1, p2, expected) in test_cases:
            params1 = {'monotone_constraints': p1} if p1 is not None else {}
            d1 = lgb.Dataset(X[:, 0].reshape((-1, 1)), params=params1).construct()
            params2 = {'monotone_constraints': p2} if p2 is not None else {}
            d2 = lgb.Dataset(X[:, 1].reshape((-1, 1)), params=params2).construct()
            d1.add_features_from(d2)
            actual = d1.get_monotone_constraints()
            if actual is None:
                self.assertIsNone(actual)
            else:
                np.testing.assert_array_equal(actual, expected)

    def test_cegb_affects_behavior(self):
        X = np.random.random((1000, 5))
        X[:, [1, 3]] = 0
        y = np.random.random(1000)
        names = ['col_%d' % i for i in range(5)]
        ds = lgb.Dataset(X, feature_name=names).construct()
        ds.set_label(y)
        base = lgb.Booster(train_set=ds)
        for k in range(10):
            base.update()
        with tempfile.NamedTemporaryFile() as f:
            basename = f.name
        base.save_model(basename)
        with open(basename, 'rt') as f:
            basetxt = f.read()
        # Set extremely harsh penalties, so CEGB will block most splits.
        cases = [{'cegb_penalty_feature_coupled': [50, 100, 10, 25, 30]},
                 {'cegb_penalty_feature_lazy': [1, 2, 3, 4, 5]},
                 {'cegb_penalty_split': 1}]
        for case in cases:
            booster = lgb.Booster(train_set=ds, params=case)
            for k in range(10):
                booster.update()
            with tempfile.NamedTemporaryFile() as f:
                casename = f.name
            booster.save_model(casename)
            with open(casename, 'rt') as f:
                casetxt = f.read()
            self.assertNotEqual(basetxt, casetxt)

    def test_cegb_scaling_equalities(self):
        X = np.random.random((1000, 5))
        X[:, [1, 3]] = 0
        y = np.random.random(1000)
        names = ['col_%d' % i for i in range(5)]
        ds = lgb.Dataset(X, feature_name=names).construct()
        ds.set_label(y)
        # Compare pairs of penalties, to ensure scaling works as intended
        pairs = [({'cegb_penalty_feature_coupled': [1, 2, 1, 2, 1]},
                  {'cegb_penalty_feature_coupled': [0.5, 1, 0.5, 1, 0.5], 'cegb_tradeoff': 2}),
                 ({'cegb_penalty_feature_lazy': [0.01, 0.02, 0.03, 0.04, 0.05]},
                  {'cegb_penalty_feature_lazy': [0.005, 0.01, 0.015, 0.02, 0.025], 'cegb_tradeoff': 2}),
                 ({'cegb_penalty_split': 1},
                  {'cegb_penalty_split': 2, 'cegb_tradeoff': 0.5})]
        for (p1, p2) in pairs:
            booster1 = lgb.Booster(train_set=ds, params=p1)
            booster2 = lgb.Booster(train_set=ds, params=p2)
            for k in range(10):
                booster1.update()
                booster2.update()
            with tempfile.NamedTemporaryFile() as f:
                p1name = f.name
            # Reset booster1's parameters to p2, so the parameter section of the file matches.
            booster1.reset_parameter(p2)
            booster1.save_model(p1name)
            with open(p1name, 'rt') as f:
                p1txt = f.read()
            with tempfile.NamedTemporaryFile() as f:
                p2name = f.name
            booster2.save_model(p2name)
            self.maxDiff = None
            with open(p2name, 'rt') as f:
                p2txt = f.read()
            self.assertEqual(p1txt, p2txt)

    def test_consistent_state_for_dataset_fields(self):

        def check_asserts(data):
            np.testing.assert_allclose(data.label, data.get_label())
            np.testing.assert_allclose(data.label, data.get_field('label'))
            self.assertFalse(np.isnan(data.label[0]))
            self.assertFalse(np.isinf(data.label[1]))
            np.testing.assert_allclose(data.weight, data.get_weight())
            np.testing.assert_allclose(data.weight, data.get_field('weight'))
            self.assertFalse(np.isnan(data.weight[0]))
            self.assertFalse(np.isinf(data.weight[1]))
            np.testing.assert_allclose(data.init_score, data.get_init_score())
            np.testing.assert_allclose(data.init_score, data.get_field('init_score'))
            self.assertFalse(np.isnan(data.init_score[0]))
            self.assertFalse(np.isinf(data.init_score[1]))
            self.assertTrue(np.all(np.isclose([data.label[0], data.weight[0], data.init_score[0]],
                                              data.label[0])))
            self.assertAlmostEqual(data.label[1], data.weight[1])

        X, y = load_breast_cancer(True)
        sequence = np.ones(y.shape[0])
        sequence[0] = np.nan
        sequence[1] = np.inf
        lgb_data = lgb.Dataset(X, sequence, weight=sequence, init_score=sequence).construct()
        check_asserts(lgb_data)
        lgb_data = lgb.Dataset(X, y).construct()
        lgb_data.set_label(sequence)
        lgb_data.set_weight(sequence)
        lgb_data.set_init_score(sequence)
        check_asserts(lgb_data)
