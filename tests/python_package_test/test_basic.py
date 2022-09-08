# coding: utf-8
import filecmp
import numbers
import re
from os import getenv
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
from sklearn.datasets import dump_svmlight_file, load_svmlight_file
from sklearn.model_selection import train_test_split

import lightgbm as lgb
from lightgbm.compat import PANDAS_INSTALLED, pd_DataFrame, pd_Series

from .utils import dummy_obj, load_breast_cancer, mse_obj


def test_basic(tmp_path):
    X_train, X_test, y_train, y_test = train_test_split(*load_breast_cancer(return_X_y=True),
                                                        test_size=0.1, random_state=2)
    feature_names = [f"Column_{i}" for i in range(X_train.shape[1])]
    feature_names[1] = "a" * 1000  # set one name to a value longer than default buffer size
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    valid_data = train_data.create_valid(X_test, label=y_test)

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

    assert train_data.get_feature_name() == feature_names

    assert bst.current_iteration() == 20
    assert bst.num_trees() == 20
    assert bst.num_model_per_iteration() == 1
    if getenv('TASK', '') != 'cuda_exp':
        assert bst.lower_bound() == pytest.approx(-2.9040190126976606)
        assert bst.upper_bound() == pytest.approx(3.3182142872462883)

    tname = tmp_path / "svm_light.dat"
    model_file = tmp_path / "model.txt"

    bst.save_model(model_file)
    pred_from_matr = bst.predict(X_test)
    with open(tname, "w+b") as f:
        dump_svmlight_file(X_test, y_test, f)
    pred_from_file = bst.predict(tname)
    np.testing.assert_allclose(pred_from_matr, pred_from_file)

    # check saved model persistence
    bst = lgb.Booster(params, model_file=model_file)
    assert bst.feature_name() == feature_names
    pred_from_model_file = bst.predict(X_test)
    # we need to check the consistency of model file here, so test for exact equal
    np.testing.assert_array_equal(pred_from_matr, pred_from_model_file)

    # check early stopping is working. Make it stop very early, so the scores should be very close to zero
    pred_parameter = {"pred_early_stop": True, "pred_early_stop_freq": 5, "pred_early_stop_margin": 1.5}
    pred_early_stopping = bst.predict(X_test, **pred_parameter)
    # scores likely to be different, but prediction should still be the same
    np.testing.assert_array_equal(np.sign(pred_from_matr), np.sign(pred_early_stopping))

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


class NumpySequence(lgb.Sequence):
    def __init__(self, ndarray, batch_size):
        self.ndarray = ndarray
        self.batch_size = batch_size

    def __getitem__(self, idx):
        # The simple implementation is just a single "return self.ndarray[idx]"
        # The following is for demo and testing purpose.
        if isinstance(idx, numbers.Integral):
            return self.ndarray[idx]
        elif isinstance(idx, slice):
            if not (idx.step is None or idx.step == 1):
                raise NotImplementedError("No need to implement, caller will not set step by now")
            return self.ndarray[idx.start:idx.stop]
        elif isinstance(idx, list):
            return self.ndarray[idx]
        else:
            raise TypeError(f"Sequence Index must be an integer/list/slice, got {type(idx).__name__}")

    def __len__(self):
        return len(self.ndarray)


def _create_sequence_from_ndarray(data, num_seq, batch_size):
    if num_seq == 1:
        return NumpySequence(data, batch_size)

    nrow = data.shape[0]
    seqs = []
    seq_size = nrow // num_seq
    for start in range(0, nrow, seq_size):
        end = min(start + seq_size, nrow)
        seq = NumpySequence(data[start:end], batch_size)
        seqs.append(seq)
    return seqs


@pytest.mark.parametrize('sample_count', [11, 100, None])
@pytest.mark.parametrize('batch_size', [3, None])
@pytest.mark.parametrize('include_0_and_nan', [False, True])
@pytest.mark.parametrize('num_seq', [1, 3])
def test_sequence(tmpdir, sample_count, batch_size, include_0_and_nan, num_seq):
    params = {'bin_construct_sample_cnt': sample_count}

    nrow = 50
    half_nrow = nrow // 2
    ncol = 11
    data = np.arange(nrow * ncol, dtype=np.float64).reshape((nrow, ncol))

    if include_0_and_nan:
        # whole col
        data[:, 0] = 0
        data[:, 1] = np.nan

        # half col
        data[:half_nrow, 3] = 0
        data[:half_nrow, 2] = np.nan

        data[half_nrow:-2, 4] = 0
        data[:half_nrow, 4] = np.nan

    X = data[:, :-1]
    Y = data[:, -1]

    npy_bin_fname = tmpdir / 'data_from_npy.bin'
    seq_bin_fname = tmpdir / 'data_from_seq.bin'

    # Create dataset from numpy array directly.
    ds = lgb.Dataset(X, label=Y, params=params)
    ds.save_binary(npy_bin_fname)

    # Create dataset using Sequence.
    seqs = _create_sequence_from_ndarray(X, num_seq, batch_size)
    seq_ds = lgb.Dataset(seqs, label=Y, params=params)
    seq_ds.save_binary(seq_bin_fname)

    assert filecmp.cmp(npy_bin_fname, seq_bin_fname)

    # Test for validation set.
    # Select some random rows as valid data.
    rng = np.random.default_rng()  # Pass integer to set seed when needed.
    valid_idx = (rng.random(10) * nrow).astype(np.int32)
    valid_data = data[valid_idx, :]
    valid_X = valid_data[:, :-1]
    valid_Y = valid_data[:, -1]

    valid_npy_bin_fname = tmpdir / 'valid_data_from_npy.bin'
    valid_seq_bin_fname = tmpdir / 'valid_data_from_seq.bin'
    valid_seq2_bin_fname = tmpdir / 'valid_data_from_seq2.bin'

    valid_ds = lgb.Dataset(valid_X, label=valid_Y, params=params, reference=ds)
    valid_ds.save_binary(valid_npy_bin_fname)

    # From Dataset constructor, with dataset from numpy array.
    valid_seqs = _create_sequence_from_ndarray(valid_X, num_seq, batch_size)
    valid_seq_ds = lgb.Dataset(valid_seqs, label=valid_Y, params=params, reference=ds)
    valid_seq_ds.save_binary(valid_seq_bin_fname)
    assert filecmp.cmp(valid_npy_bin_fname, valid_seq_bin_fname)

    # From Dataset.create_valid, with dataset from sequence.
    valid_seq_ds2 = seq_ds.create_valid(valid_seqs, label=valid_Y, params=params)
    valid_seq_ds2.save_binary(valid_seq2_bin_fname)
    assert filecmp.cmp(valid_npy_bin_fname, valid_seq2_bin_fname)


@pytest.mark.parametrize('num_seq', [1, 2])
def test_sequence_get_data(num_seq):
    nrow = 20
    ncol = 11
    data = np.arange(nrow * ncol, dtype=np.float64).reshape((nrow, ncol))
    X = data[:, :-1]
    Y = data[:, -1]

    seqs = _create_sequence_from_ndarray(data=X, num_seq=num_seq, batch_size=6)
    seq_ds = lgb.Dataset(seqs, label=Y, params=None, free_raw_data=False).construct()
    assert seq_ds.get_data() == seqs

    used_indices = np.random.choice(np.arange(nrow), nrow // 3, replace=False)
    subset_data = seq_ds.subset(used_indices).construct()
    np.testing.assert_array_equal(subset_data.get_data(), X[sorted(used_indices)])


def test_chunked_dataset():
    X_train, X_test, y_train, y_test = train_test_split(*load_breast_cancer(return_X_y=True), test_size=0.1,
                                                        random_state=2)

    chunk_size = X_train.shape[0] // 10 + 1
    X_train = [X_train[i * chunk_size:(i + 1) * chunk_size, :] for i in range(X_train.shape[0] // chunk_size + 1)]
    X_test = [X_test[i * chunk_size:(i + 1) * chunk_size, :] for i in range(X_test.shape[0] // chunk_size + 1)]

    train_data = lgb.Dataset(X_train, label=y_train, params={"bin_construct_sample_cnt": 100})
    valid_data = train_data.create_valid(X_test, label=y_test, params={"bin_construct_sample_cnt": 100})
    train_data.construct()
    valid_data.construct()


def test_chunked_dataset_linear():
    X_train, X_test, y_train, y_test = train_test_split(*load_breast_cancer(return_X_y=True), test_size=0.1,
                                                        random_state=2)
    chunk_size = X_train.shape[0] // 10 + 1
    X_train = [X_train[i * chunk_size:(i + 1) * chunk_size, :] for i in range(X_train.shape[0] // chunk_size + 1)]
    X_test = [X_test[i * chunk_size:(i + 1) * chunk_size, :] for i in range(X_test.shape[0] // chunk_size + 1)]
    params = {"bin_construct_sample_cnt": 100, 'linear_tree': True}
    train_data = lgb.Dataset(X_train, label=y_train, params=params)
    valid_data = train_data.create_valid(X_test, label=y_test, params=params)
    train_data.construct()
    valid_data.construct()


def test_save_dataset_subset_and_load_from_file(tmp_path):
    data = np.random.rand(100, 2)
    params = {'max_bin': 50, 'min_data_in_bin': 10}
    ds = lgb.Dataset(data, params=params)
    ds.subset([1, 2, 3, 5, 8]).save_binary(tmp_path / 'subset.bin')
    lgb.Dataset(tmp_path / 'subset.bin', params=params).construct()


def test_subset_group():
    rank_example_dir = Path(__file__).absolute().parents[2] / 'examples' / 'lambdarank'
    X_train, y_train = load_svmlight_file(str(rank_example_dir / 'rank.train'))
    q_train = np.loadtxt(str(rank_example_dir / 'rank.train.query'))
    lgb_train = lgb.Dataset(X_train, y_train, group=q_train)
    assert len(lgb_train.get_group()) == 201
    subset = lgb_train.subset(list(range(10))).construct()
    subset_group = subset.get_group()
    assert len(subset_group) == 2
    assert subset_group[0] == 1
    assert subset_group[1] == 9


def test_add_features_throws_if_num_data_unequal():
    X1 = np.random.random((100, 1))
    X2 = np.random.random((10, 1))
    d1 = lgb.Dataset(X1).construct()
    d2 = lgb.Dataset(X2).construct()
    with pytest.raises(lgb.basic.LightGBMError):
        d1.add_features_from(d2)


def test_add_features_throws_if_datasets_unconstructed():
    X1 = np.random.random((100, 1))
    X2 = np.random.random((100, 1))
    with pytest.raises(ValueError):
        d1 = lgb.Dataset(X1)
        d2 = lgb.Dataset(X2)
        d1.add_features_from(d2)
    with pytest.raises(ValueError):
        d1 = lgb.Dataset(X1).construct()
        d2 = lgb.Dataset(X2)
        d1.add_features_from(d2)
    with pytest.raises(ValueError):
        d1 = lgb.Dataset(X1)
        d2 = lgb.Dataset(X2).construct()
        d1.add_features_from(d2)


def test_add_features_equal_data_on_alternating_used_unused(tmp_path):
    X = np.random.random((100, 5))
    X[:, [1, 3]] = 0
    names = [f'col_{i}' for i in range(5)]
    for j in range(1, 5):
        d1 = lgb.Dataset(X[:, :j], feature_name=names[:j]).construct()
        d2 = lgb.Dataset(X[:, j:], feature_name=names[j:]).construct()
        d1.add_features_from(d2)
        d1name = tmp_path / "d1.txt"
        d1._dump_text(d1name)
        d = lgb.Dataset(X, feature_name=names).construct()
        dname = tmp_path / "d.txt"
        d._dump_text(dname)
        with open(d1name, 'rt') as d1f:
            d1txt = d1f.read()
        with open(dname, 'rt') as df:
            dtxt = df.read()
        assert dtxt == d1txt


def test_add_features_same_booster_behaviour(tmp_path):
    X = np.random.random((100, 5))
    X[:, [1, 3]] = 0
    names = [f'col_{i}' for i in range(5)]
    for j in range(1, 5):
        d1 = lgb.Dataset(X[:, :j], feature_name=names[:j]).construct()
        d2 = lgb.Dataset(X[:, j:], feature_name=names[j:]).construct()
        d1.add_features_from(d2)
        d = lgb.Dataset(X, feature_name=names).construct()
        y = np.random.random(100)
        d1.set_label(y)
        d.set_label(y)
        b1 = lgb.Booster(train_set=d1)
        b = lgb.Booster(train_set=d)
        for k in range(10):
            b.update()
            b1.update()
        dname = tmp_path / "d.txt"
        d1name = tmp_path / "d1.txt"
        b1.save_model(d1name)
        b.save_model(dname)
        with open(dname, 'rt') as df:
            dtxt = df.read()
        with open(d1name, 'rt') as d1f:
            d1txt = d1f.read()
        assert dtxt == d1txt


def test_add_features_from_different_sources():
    pd = pytest.importorskip("pandas")
    n_row = 100
    n_col = 5
    X = np.random.random((n_row, n_col))
    xxs = [X, sparse.csr_matrix(X), pd.DataFrame(X)]
    names = [f'col_{i}' for i in range(n_col)]
    seq = _create_sequence_from_ndarray(X, 1, 30)
    seq_ds = lgb.Dataset(seq, feature_name=names, free_raw_data=False).construct()
    npy_list_ds = lgb.Dataset([X[:n_row // 2, :], X[n_row // 2:, :]],
                              feature_name=names, free_raw_data=False).construct()
    immergeable_dds = [seq_ds, npy_list_ds]
    for x_1 in xxs:
        # test that method works even with free_raw_data=True
        d1 = lgb.Dataset(x_1, feature_name=names, free_raw_data=True).construct()
        d2 = lgb.Dataset(x_1, feature_name=names, free_raw_data=True).construct()
        d1.add_features_from(d2)
        assert d1.data is None

        # test that method works but sets raw data to None in case of immergeable data types
        d1 = lgb.Dataset(x_1, feature_name=names, free_raw_data=False).construct()
        for d2 in immergeable_dds:
            d1.add_features_from(d2)
            assert d1.data is None

        # test that method works for different data types
        d1 = lgb.Dataset(x_1, feature_name=names, free_raw_data=False).construct()
        res_feature_names = [name for name in names]
        for idx, x_2 in enumerate(xxs, 2):
            original_type = type(d1.get_data())
            d2 = lgb.Dataset(x_2, feature_name=names, free_raw_data=False).construct()
            d1.add_features_from(d2)
            assert isinstance(d1.get_data(), original_type)
            assert d1.get_data().shape == (n_row, n_col * idx)
            res_feature_names += [f'D{idx}_{name}' for name in names]
            assert d1.feature_name == res_feature_names


def test_cegb_affects_behavior(tmp_path):
    X = np.random.random((100, 5))
    X[:, [1, 3]] = 0
    y = np.random.random(100)
    names = [f'col_{i}' for i in range(5)]
    ds = lgb.Dataset(X, feature_name=names).construct()
    ds.set_label(y)
    base = lgb.Booster(train_set=ds)
    for k in range(10):
        base.update()
    basename = tmp_path / "basename.txt"
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
        casename = tmp_path / "casename.txt"
        booster.save_model(casename)
        with open(casename, 'rt') as f:
            casetxt = f.read()
        assert basetxt != casetxt


def test_cegb_scaling_equalities(tmp_path):
    X = np.random.random((100, 5))
    X[:, [1, 3]] = 0
    y = np.random.random(100)
    names = [f'col_{i}' for i in range(5)]
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
        p1name = tmp_path / "p1.txt"
        # Reset booster1's parameters to p2, so the parameter section of the file matches.
        booster1.reset_parameter(p2)
        booster1.save_model(p1name)
        with open(p1name, 'rt') as f:
            p1txt = f.read()
        p2name = tmp_path / "p2.txt"
        booster2.save_model(p2name)
        with open(p2name, 'rt') as f:
            p2txt = f.read()
        assert p1txt == p2txt


def test_consistent_state_for_dataset_fields():

    def check_asserts(data):
        np.testing.assert_allclose(data.label, data.get_label())
        np.testing.assert_allclose(data.label, data.get_field('label'))
        assert not np.isnan(data.label[0])
        assert not np.isinf(data.label[1])
        np.testing.assert_allclose(data.weight, data.get_weight())
        np.testing.assert_allclose(data.weight, data.get_field('weight'))
        assert not np.isnan(data.weight[0])
        assert not np.isinf(data.weight[1])
        np.testing.assert_allclose(data.init_score, data.get_init_score())
        np.testing.assert_allclose(data.init_score, data.get_field('init_score'))
        assert not np.isnan(data.init_score[0])
        assert not np.isinf(data.init_score[1])
        assert np.all(np.isclose([data.label[0], data.weight[0], data.init_score[0]],
                                 data.label[0]))
        assert data.label[1] == pytest.approx(data.weight[1])
        assert data.feature_name == data.get_feature_name()

    X, y = load_breast_cancer(return_X_y=True)
    sequence = np.ones(y.shape[0])
    sequence[0] = np.nan
    sequence[1] = np.inf
    feature_names = [f'f{i}'for i in range(X.shape[1])]
    lgb_data = lgb.Dataset(X, sequence,
                           weight=sequence, init_score=sequence,
                           feature_name=feature_names).construct()
    check_asserts(lgb_data)
    lgb_data = lgb.Dataset(X, y).construct()
    lgb_data.set_label(sequence)
    lgb_data.set_weight(sequence)
    lgb_data.set_init_score(sequence)
    lgb_data.set_feature_name(feature_names)
    check_asserts(lgb_data)


def test_choose_param_value():

    original_params = {
        "local_listen_port": 1234,
        "port": 2222,
        "metric": "auc",
        "num_trees": 81,
        "n_iter": 13,
    }

    # should resolve duplicate aliases, and prefer the main parameter
    params = lgb.basic._choose_param_value(
        main_param_name="local_listen_port",
        params=original_params,
        default_value=5555
    )
    assert params["local_listen_port"] == 1234
    assert "port" not in params

    # should choose the highest priority alias and set that value on main param
    # if only aliases are used
    params = lgb.basic._choose_param_value(
        main_param_name="num_iterations",
        params=params,
        default_value=17
    )
    assert params["num_iterations"] == 13
    assert "num_trees" not in params
    assert "n_iter" not in params

    # should use the default if main param and aliases are missing
    params = lgb.basic._choose_param_value(
        main_param_name="learning_rate",
        params=params,
        default_value=0.789
    )
    assert params["learning_rate"] == 0.789

    # all changes should be made on copies and not modify the original
    expected_params = {
        "local_listen_port": 1234,
        "port": 2222,
        "metric": "auc",
        "num_trees": 81,
        "n_iter": 13,
    }
    assert original_params == expected_params


def test_choose_param_value_preserves_nones():

    # preserves None found for main param and still removes aliases
    params = lgb.basic._choose_param_value(
        main_param_name="num_threads",
        params={
            "num_threads": None,
            "n_jobs": 4,
            "objective": "regression"
        },
        default_value=2
    )
    assert params == {"num_threads": None, "objective": "regression"}

    # correctly chooses value when only an alias is provided
    params = lgb.basic._choose_param_value(
        main_param_name="num_threads",
        params={
            "n_jobs": None,
            "objective": "regression"
        },
        default_value=2
    )
    assert params == {"num_threads": None, "objective": "regression"}

    # adds None if that's given as the default and param not found
    params = lgb.basic._choose_param_value(
        main_param_name="min_data_in_leaf",
        params={
            "objective": "regression"
        },
        default_value=None
    )
    assert params == {"objective": "regression", "min_data_in_leaf": None}


@pytest.mark.parametrize("objective_alias", lgb.basic._ConfigAliases.get("objective"))
def test_choose_param_value_objective(objective_alias):
    # If callable is found in objective
    params = {objective_alias: dummy_obj}
    params = lgb.basic._choose_param_value(
        main_param_name="objective",
        params=params,
        default_value=None
    )
    assert params['objective'] == dummy_obj

    # Value in params should be preferred to the default_value passed from keyword arguments
    params = {objective_alias: dummy_obj}
    params = lgb.basic._choose_param_value(
        main_param_name="objective",
        params=params,
        default_value=mse_obj
    )
    assert params['objective'] == dummy_obj

    # None of objective or its aliases in params, but default_value is callable.
    params = {}
    params = lgb.basic._choose_param_value(
        main_param_name="objective",
        params=params,
        default_value=mse_obj
    )
    assert params['objective'] == mse_obj


@pytest.mark.parametrize('collection', ['1d_np', '2d_np', 'pd_float', 'pd_str', '1d_list', '2d_list'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_list_to_1d_numpy(collection, dtype):
    collection2y = {
        '1d_np': np.random.rand(10),
        '2d_np': np.random.rand(10, 1),
        'pd_float': np.random.rand(10),
        'pd_str': ['a', 'b'],
        '1d_list': [1] * 10,
        '2d_list': [[1], [2]],
    }
    y = collection2y[collection]
    if collection.startswith('pd'):
        if not PANDAS_INSTALLED:
            pytest.skip('pandas is not installed')
        else:
            y = pd_Series(y)
    if isinstance(y, np.ndarray) and len(y.shape) == 2:
        with pytest.warns(UserWarning, match='column-vector'):
            lgb.basic.list_to_1d_numpy(y)
        return
    elif isinstance(y, list) and isinstance(y[0], list):
        with pytest.raises(TypeError):
            lgb.basic.list_to_1d_numpy(y)
        return
    elif isinstance(y, pd_Series) and y.dtype == object:
        with pytest.raises(ValueError):
            lgb.basic.list_to_1d_numpy(y)
        return
    result = lgb.basic.list_to_1d_numpy(y, dtype=dtype)
    assert result.size == 10
    assert result.dtype == dtype


@pytest.mark.parametrize('init_score_type', ['array', 'dataframe', 'list'])
def test_init_score_for_multiclass_classification(init_score_type):
    init_score = [[i * 10 + j for j in range(3)] for i in range(10)]
    if init_score_type == 'array':
        init_score = np.array(init_score)
    elif init_score_type == 'dataframe':
        if not PANDAS_INSTALLED:
            pytest.skip('Pandas is not installed.')
        init_score = pd_DataFrame(init_score)
    data = np.random.rand(10, 2)
    ds = lgb.Dataset(data, init_score=init_score).construct()
    np.testing.assert_equal(ds.get_field('init_score'), init_score)
    np.testing.assert_equal(ds.init_score, init_score)


def test_smoke_custom_parser(tmp_path):
    data_path = Path(__file__).absolute().parents[2] / 'examples' / 'binary_classification' / 'binary.train'
    parser_config_file = tmp_path / 'parser.ini'
    with open(parser_config_file, 'w') as fout:
        fout.write('{"className": "dummy", "id": "1"}')

    data = lgb.Dataset(data_path, params={"parser_config_file": parser_config_file})
    with pytest.raises(lgb.basic.LightGBMError,
                       match="Cannot find parser class 'dummy', please register first or check config format"):
        data.construct()


def test_param_aliases():
    aliases = lgb.basic._ConfigAliases.aliases
    assert isinstance(aliases, dict)
    assert len(aliases) > 100
    assert all(isinstance(i, list) for i in aliases.values())
    assert all(len(i) >= 1 for i in aliases.values())
    assert all(k in v for k, v in aliases.items())
    assert lgb.basic._ConfigAliases.get('config', 'task') == {'config', 'config_file', 'task', 'task_type'}
    assert lgb.basic._ConfigAliases.get_sorted('min_data_in_leaf') == [
        'min_data_in_leaf', 'min_data', 'min_samples_leaf', 'min_child_samples', 'min_data_per_leaf'
    ]


def _bad_gradients(preds, _):
    return np.random.randn(len(preds) + 1), np.random.rand(len(preds) + 1)


def _good_gradients(preds, _):
    return np.random.randn(*preds.shape), np.random.rand(*preds.shape)


def test_custom_objective_safety():
    nrows = 100
    X = np.random.randn(nrows, 5)
    y_binary = np.arange(nrows) % 2
    classes = [0, 1, 2]
    nclass = len(classes)
    y_multiclass = np.arange(nrows) % nclass
    ds_binary = lgb.Dataset(X, y_binary).construct()
    ds_multiclass = lgb.Dataset(X, y_multiclass).construct()
    bad_bst_binary = lgb.Booster({'objective': "none"}, ds_binary)
    good_bst_binary = lgb.Booster({'objective': "none"}, ds_binary)
    bad_bst_multi = lgb.Booster({'objective': "none", "num_class": nclass}, ds_multiclass)
    good_bst_multi = lgb.Booster({'objective': "none", "num_class": nclass}, ds_multiclass)
    good_bst_binary.update(fobj=_good_gradients)
    with pytest.raises(ValueError, match=re.escape("number of models per one iteration (1)")):
        bad_bst_binary.update(fobj=_bad_gradients)
    good_bst_multi.update(fobj=_good_gradients)
    with pytest.raises(ValueError, match=re.escape(f"number of models per one iteration ({nclass})")):
        bad_bst_multi.update(fobj=_bad_gradients)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('feature_name', [['x1', 'x2'], 'auto'])
def test_no_copy_when_single_float_dtype_dataframe(dtype, feature_name):
    pd = pytest.importorskip('pandas')
    X = np.random.rand(10, 2).astype(dtype)
    df = pd.DataFrame(X)
    built_data = lgb.basic._data_from_pandas(df, feature_name, None, None)[0]
    assert built_data.dtype == dtype
    assert np.shares_memory(X, built_data)


@pytest.mark.parametrize('feature_name', [['x1'], [42], 'auto'])
def test_categorical_code_conversion_doesnt_modify_original_data(feature_name):
    pd = pytest.importorskip('pandas')
    X = np.random.choice(['a', 'b'], 100).reshape(-1, 1)
    column_name = 'a' if feature_name == 'auto' else feature_name[0]
    df = pd.DataFrame(X.copy(), columns=[column_name], dtype='category')
    data = lgb.basic._data_from_pandas(df, feature_name, None, None)[0]
    # check that the original data wasn't modified
    np.testing.assert_equal(df[column_name], X[:, 0])
    # check that the built data has the codes
    np.testing.assert_equal(df[column_name].cat.codes, data[:, 0])


@pytest.mark.parametrize('min_data_in_bin', [2, 10])
def test_feature_num_bin(min_data_in_bin):
    X = np.vstack([
        np.random.rand(100),
        np.array([1, 2] * 50),
        np.array([0, 1, 2] * 33 + [0]),
        np.array([1, 2] * 49 + 2 * [np.nan]),
        np.zeros(100),
        np.random.choice([0, 1], 100),
    ]).T
    n_continuous = X.shape[1] - 1
    feature_name = [f'x{i}' for i in range(n_continuous)] + ['cat1']
    ds_kwargs = dict(
        params={'min_data_in_bin': min_data_in_bin},
        categorical_feature=[n_continuous],  # last feature
    )
    ds = lgb.Dataset(X, feature_name=feature_name, **ds_kwargs).construct()
    expected_num_bins = [
        100 // min_data_in_bin + 1,  # extra bin for zero
        3,  # 0, 1, 2
        3,  # 0, 1, 2
        4,  # 0, 1, 2 + nan
        0,  # unused
        3,  # 0, 1 + nan
    ]
    actual_num_bins = [ds.feature_num_bin(i) for i in range(X.shape[1])]
    assert actual_num_bins == expected_num_bins
    # test using defined feature names
    bins_by_name = [ds.feature_num_bin(name) for name in feature_name]
    assert bins_by_name == expected_num_bins
    # test using default feature names
    ds_no_names = lgb.Dataset(X, **ds_kwargs).construct()
    default_names = [f'Column_{i}' for i in range(X.shape[1])]
    bins_by_default_name = [ds_no_names.feature_num_bin(name) for name in default_names]
    assert bins_by_default_name == expected_num_bins
    # check for feature indices outside of range
    num_features = X.shape[1]
    with pytest.raises(
        lgb.basic.LightGBMError,
        match=(
            f'Tried to retrieve number of bins for feature index {num_features}, '
            f'but the valid feature indices are \\[0, {num_features - 1}\\].'
        )
    ):
        ds.feature_num_bin(num_features)


def test_feature_num_bin_with_max_bin_by_feature():
    X = np.random.rand(100, 3)
    max_bin_by_feature = np.random.randint(3, 30, size=X.shape[1])
    ds = lgb.Dataset(X, params={'max_bin_by_feature': max_bin_by_feature}).construct()
    actual_num_bins = [ds.feature_num_bin(i) for i in range(X.shape[1])]
    np.testing.assert_equal(actual_num_bins, max_bin_by_feature)
