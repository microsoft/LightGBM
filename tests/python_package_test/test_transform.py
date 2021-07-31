from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import lightgbm as lgb

Dataset = namedtuple('Dataset', ['data', 'transform', 'header'])


def has_transform_lib_installed():
    lgb_path = Path(lgb.__file__).parent
    pkg_path = lgb_path.parent
    # get from PYTHONPATH or site-packages.
    dll_path = pkg_path.parent if pkg_path.name == 'python-package' else lgb_path
    return len(list(dll_path.glob('**/lib_transform.so'))) > 0


def root_path():
    return Path(__file__).parent.parent / "data"


def init_dataset(data_dir):
    return Dataset(data_dir / "input.tsv", data_dir / "transform.ini", data_dir / "header.tsv")


rank_ds = init_dataset(root_path() / "transform_rank_data")
simple_ds = init_dataset(root_path() / "transform_simple_data")


@pytest.fixture
def rank_data_with_header(tmp_path):
    out_path = tmp_path / "rank_input_with_header.tsv"
    hdf = pd.read_csv(rank_ds.header, sep='\t')
    df = pd.read_csv(rank_ds.data, sep='\t', header=None)
    df.to_csv(out_path, index=False, header=list(hdf.columns), sep='\t')
    return out_path


@pytest.fixture
def simple_data_with_header(tmp_path):
    out_path = tmp_path / "simple_input_with_header.tsv"
    hdf = pd.read_csv(simple_ds.header, sep='\t')
    df = pd.read_csv(simple_ds.data, sep='\t', header=None)
    df.to_csv(out_path, index=False, header=list(hdf.columns), sep='\t')
    return out_path


@pytest.fixture
def simple_data_no_label(simple_data_with_header, tmp_path):
    out_path = tmp_path / "simple_input_no_label.tsv"
    df = pd.read_csv(simple_data_with_header, sep='\t', header=0)
    df.drop("labels", axis=1).to_csv(out_path, index=False, sep='\t')
    return out_path


@pytest.fixture
def trained_model_path(tmp_path):
    return tmp_path / "model.txt"


def check_if_save_in_model(model_file, input_file, section_name):
    expected_lines = []
    with open(input_file) as fin:
        for line in fin:
            if line.strip() == "":
                continue
            expected_lines.append(line)

    is_in_sec = False
    check = True
    idx = 0
    with open(model_file) as fin:
        for line in fin:
            if line.strip() == "":
                continue
            if line.startswith(section_name):
                is_in_sec = True
                idx = 0
            elif line.startswith(f"end of {section_name}"):
                break
            elif is_in_sec:
                if line != expected_lines[idx]:
                    check = False
                    break
                idx += 1
    return len(expected_lines) == idx and check


@pytest.fixture
def params():
    return {
        'boosting': 'gbdt',
        'learning_rate': 0.1,
        'label': 3,
        'query': 0,
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'num_trees': 10,
        'num_leaves': 31,
        'label_gain': ','.join([str(i) for i in range(101)]),
        'force_col_wise': True,
        'deterministic': True
    }


@pytest.fixture
def binary_params():
    return {
        'boosting': 'gbdt',
        'learning_rate': 0.1,
        'label': 10,
        'objective': 'binary',
        'metric': 'auc',
        'num_trees': 10,
        'num_leaves': 31,
        'deterministic': True
    }


@pytest.mark.skipif(not has_transform_lib_installed(), reason="requires transform lib to be installed")
def test_e2e(params, rank_data_with_header, trained_model_path, capsys):
    train_data = lgb.Dataset(rank_ds.data, params={"transform_file": rank_ds.transform, "header_file": rank_ds.header})
    valid_data = lgb.Dataset(rank_data_with_header, params={"transform_file": rank_ds.transform, "header": True})
    # train and predict.
    bst = lgb.train(params, train_data, valid_sets=[valid_data])
    pred = bst.predict(rank_ds.data)
    np.testing.assert_allclose(pred[:5], np.array([0.83267298, 0.388454, 0.35369267, 0.60330376, -1.24218415]))
    # save model.
    bst.save_model(trained_model_path)
    captured = capsys.readouterr()
    assert "Initialize transform time" in captured.out
    assert check_if_save_in_model(trained_model_path, rank_ds.transform, "transform")
    assert check_if_save_in_model(trained_model_path, rank_ds.header, "header")
    # load model and predict again.
    bst = lgb.Booster(model_file=trained_model_path)
    pred = bst.predict(rank_ds.data)
    captured = capsys.readouterr()
    assert "Initialize transform time" in captured.out
    np.testing.assert_allclose(pred[:5], np.array([0.83267298, 0.388454, 0.35369267, 0.60330376, -1.24218415]))


@pytest.mark.skipif(not has_transform_lib_installed(), reason="requires transform lib to be installed")
@pytest.mark.parametrize("data_params, expected_pred", [
    ({"transform_file": simple_ds.transform, "header_file": simple_ds.header}, np.array([0.4894574])),
    ({"header_file": simple_ds.header}, np.array([0.58281511])),
    ({}, np.array([0.58281511])),
])
def test_train_data_no_header(binary_params, simple_data_with_header, data_params, expected_pred):
    train_data = lgb.Dataset(simple_ds.data, params=data_params)
    bst = lgb.train(binary_params, train_data, valid_sets=[train_data])
    # predict data with no header.
    pred = bst.predict(simple_ds.data)
    np.testing.assert_allclose(pred[:1], expected_pred)
    # predict data with header.
    if len(data_params) != 0:
        pred = bst.predict(simple_data_with_header, data_has_header=True)
        np.testing.assert_allclose(pred[:1], expected_pred)


@pytest.mark.skipif(not has_transform_lib_installed(), reason="requires transform lib to be installed")
@pytest.mark.parametrize("data_params, expected_pred", [
    ({"transform_file": simple_ds.transform, "header_file": simple_ds.header, "header": True}, np.array([0.4894574])),
    ({"transform_file": simple_ds.transform, "header": True}, np.array([0.4894574])),
    ({"header_file": simple_ds.header, "header": True}, np.array([0.58281511])),
    ({"header": True}, np.array([0.58281511])),
])
def test_train_data_with_header(binary_params, simple_data_with_header, data_params, expected_pred):
    train_data = lgb.Dataset(simple_data_with_header, params=data_params)
    bst = lgb.train(binary_params, train_data, valid_sets=[train_data])
    # predict data with no header.
    pred = bst.predict(simple_ds.data)
    np.testing.assert_allclose(pred[:1], expected_pred)
    # predict data with header.
    pred = bst.predict(simple_data_with_header, data_has_header=True)
    np.testing.assert_allclose(pred[:1], expected_pred)


@pytest.mark.parametrize("data_params", [
    {"transform_file": simple_ds.transform},
    {"header_file": simple_ds.header}
])
@pytest.mark.skipif(not has_transform_lib_installed(), reason="requires transform lib to be installed")
def test_transform_or_header_not_exist(binary_params, data_params, capsys):
    train_data = lgb.Dataset(simple_ds.data, params=data_params)
    lgb.train(binary_params, train_data, valid_sets=[train_data])
    captured = capsys.readouterr()
    assert "Found transform or header does not exist" in captured.out


@pytest.mark.skipif(not has_transform_lib_installed(), reason="requires transform lib to be installed")
def test_set_label_by_name(params, capsys):
    train_data = lgb.Dataset(rank_ds.data, params={"transform_file": rank_ds.transform, "header_file": rank_ds.header})
    params['label'] = "name:Rating"
    bst = lgb.train(params, train_data, valid_sets=[train_data])
    captured = capsys.readouterr()
    assert "Using column Rating as label" in captured.out
    pred = bst.predict(rank_ds.data)
    np.testing.assert_allclose(pred[:5], np.array([0.83267298, 0.388454, 0.35369267, 0.60330376, -1.24218415]))


@pytest.mark.skipif(not has_transform_lib_installed(), reason="requires transform lib to be installed")
def test_predict_data_no_label(simple_data_no_label, binary_params):
    train_data = lgb.Dataset(simple_ds.data,
                             params={"transform_file": simple_ds.transform, "header_file": simple_ds.header})
    bst = lgb.train(binary_params, train_data, valid_sets=[train_data])
    pred = bst.predict(simple_data_no_label, data_has_header=True)
    np.testing.assert_allclose(pred[:5], np.array([0.4894574, 0.43920928, 0.71112129, 0.43920928, 0.39602784]))


@pytest.mark.skipif(not has_transform_lib_installed(), reason="requires transform lib to be installed")
def test_train_label_id_less_than_transformed_feature_num(binary_params):
    train_data = lgb.Dataset(simple_ds.data,
                             params={"transform_file": simple_ds.transform, "header_file": simple_ds.header})
    bst = lgb.train(binary_params, train_data, valid_sets=[train_data])
    pred = bst.predict(simple_ds.data)
    np.testing.assert_allclose(pred[:5], np.array([0.4894574, 0.43920928, 0.71112129, 0.43920928, 0.39602784]))
