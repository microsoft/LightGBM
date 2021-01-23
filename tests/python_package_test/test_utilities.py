# coding: utf-8
import logging

import numpy as np
import lightgbm as lgb


def test_register_logger(tmp_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s | %(message)s')
    log_filename = str(tmp_path / "LightGBM_test_logger.log")
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    lgb.register_logger(logger)

    X = np.array([[1, 2, 3],
                  [1, 2, 4],
                  [1, 2, 4],
                  [1, 2, 3]],
                 dtype=np.float32)
    y = np.array([0, 1, 1, 0])
    lgb_data = lgb.Dataset(X, y)

    eval_records = {}
    lgb.train({'objective': 'binary', 'metric': ['auc', 'binary_error']},
              lgb_data, num_boost_round=10,
              valid_sets=[lgb_data], evals_result=eval_records,
              categorical_feature=[1], early_stopping_rounds=4, verbose_eval=2)

    lgb.plot_metric(eval_records)

    expected_log = r"""
WARNING | categorical_feature in Dataset is overridden.
New categorical_feature is [1]
INFO | [LightGBM] [Warning] There are no meaningful features, as all feature values are constant.
INFO | [LightGBM] [Info] Number of positive: 2, number of negative: 2
INFO | [LightGBM] [Info] Total Bins 0
INFO | [LightGBM] [Info] Number of data points in the train set: 4, number of used features: 0
INFO | [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.500000 -> initscore=0.000000
INFO | [LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements
INFO | Training until validation scores don't improve for 4 rounds
INFO | [LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements
INFO | [2]	training's auc: 0.5	training's binary_error: 0.5
INFO | [LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements
INFO | [LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements
INFO | [4]	training's auc: 0.5	training's binary_error: 0.5
INFO | [LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements
INFO | [LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements
INFO | [6]	training's auc: 0.5	training's binary_error: 0.5
INFO | [LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements
INFO | [LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements
INFO | [8]	training's auc: 0.5	training's binary_error: 0.5
INFO | [LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements
INFO | [LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements
INFO | [10]	training's auc: 0.5	training's binary_error: 0.5
INFO | Did not meet early stopping. Best iteration is:
[1]	training's auc: 0.5	training's binary_error: 0.5
WARNING | More than one metric available, picking one to plot.
""".split()

    with open(log_filename, 'rt') as f:
        actual_log = f.read().split()

    assert actual_log == expected_log
