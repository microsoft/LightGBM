# coding: utf-8
import logging

import numpy as np
import lightgbm as lgb


def test_register_logger(tmp_path):
    logger = logging.getLogger("LightGBM")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s | %(message)s')
    log_filename = str(tmp_path / "LightGBM_test_logger.log")
    file_handler = logging.FileHandler(log_filename, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    def dummy_metric(_, __):
        logger.debug('In dummy_metric')
        return 'dummy_metric', 1, True

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
              lgb_data, num_boost_round=10, feval=dummy_metric,
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
DEBUG | In dummy_metric
INFO | Training until validation scores don't improve for 4 rounds
INFO | [LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements
DEBUG | In dummy_metric
INFO | [2]	training's auc: 0.5	training's binary_error: 0.5	training's dummy_metric: 1
INFO | [LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements
DEBUG | In dummy_metric
INFO | [LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements
DEBUG | In dummy_metric
INFO | [4]	training's auc: 0.5	training's binary_error: 0.5	training's dummy_metric: 1
INFO | [LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements
DEBUG | In dummy_metric
INFO | [LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements
DEBUG | In dummy_metric
INFO | [6]	training's auc: 0.5	training's binary_error: 0.5	training's dummy_metric: 1
INFO | [LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements
DEBUG | In dummy_metric
INFO | [LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements
DEBUG | In dummy_metric
INFO | [8]	training's auc: 0.5	training's binary_error: 0.5	training's dummy_metric: 1
INFO | [LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements
DEBUG | In dummy_metric
INFO | [LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements
DEBUG | In dummy_metric
INFO | [10]	training's auc: 0.5	training's binary_error: 0.5	training's dummy_metric: 1
INFO | Did not meet early stopping. Best iteration is:
[1]	training's auc: 0.5	training's binary_error: 0.5	training's dummy_metric: 1
WARNING | More than one metric available, picking one to plot.
""".split()

    with open(log_filename, "rt", encoding="utf-8") as f:
        actual_log = f.read().split()

    assert actual_log == expected_log
