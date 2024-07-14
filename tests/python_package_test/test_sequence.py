#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: mithril
# Created Date: 2024-07-14 14:18:46
# Last Modified: 2024-07-14 14:44:50


import numpy as np
import pytest
import sklearn.datasets
import lightgbm as lgb


class PartitionSequence(lgb.Sequence):
    def __init__(self, data:np.ndarray, batch_size=4096):
        self.data = data
        self.batch_size = batch_size

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def test_list_of_sequence():
    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    X_seq = list()
    y_seq = list()
    for i in range(2):
        X_seq.append(PartitionSequence(X, 200))
        y_seq.append(y)
        
    y = np.concatenate(y_seq)

    dataset = lgb.Dataset(X_seq, label=y, free_raw_data=False)

    params = {
        "objective": "binary",
        "metric": "auc",
        "min_data": 10,
        "num_leaves": 10,
        "verbose": -1,
        "num_threads": 1,
        "max_bin": 255,
        "gpu_use_dp": True,
    }

    model1 = lgb.train(
        params,
        dataset,
        keep_training_booster=True,
    )

    model2 = lgb.train(
        params,
        dataset,
        init_model=model1,
    )