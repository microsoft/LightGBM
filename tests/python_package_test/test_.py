import sys
sys.path = ["/Users/kenichi.matsui/Documents/996_github/LightGBM/python-package/"] + sys.path

import lightgbm as lgb
print(lgb.__file__)

import traceback
import numpy as np
import numpy.random as rd
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection._search import ParameterGrid

from IPython.display import display
from itertools import permutations


# グラフ描画系
import matplotlib
from matplotlib import font_manager
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc

from matplotlib import animation as ani
from IPython.display import Image

plt.rcParams["patch.force_edgecolor"] = True
#rc('text', usetex=True)
from IPython.display import display # Allows the use of display() for DataFrames
import seaborn as sns
sns.set(style="whitegrid", palette="muted", color_codes=True)
sns.set_style("whitegrid", {'grid.linestyle': '--'})
red = sns.xkcd_rgb["light red"]
green = sns.xkcd_rgb["medium green"]
blue = sns.xkcd_rgb["denim blue"]


##############################################################################
# Binary Classification
def load_data_for_classification():
    rd.seed(123)
    print('Loading data...')
    # load or create your dataset
    # df_train = pd.read_csv('../dataset/binary.train', header=None, sep='\t')
    # df_test = pd.read_csv('../dataset/binary.test', header=None, sep='\t')
    df_train = pd.read_csv('../../examples/binary_classification/binary.train', header=None, sep='\t')
    df_test = pd.read_csv('../../examples/binary_classification/binary.test', header=None, sep='\t')

    print("df_train.shape:{}, df_test.shape:{}".format(df_train.shape, df_test.shape))
    display(df_train.head())

    y_train = df_train[0]
    y_test = df_test[0]
    X_train = df_train.drop(0, axis=1)
    X_test = df_test.drop(0, axis=1)
    return X_train, y_train, X_test, y_test


def classification_train(dataset, metric_list=None, first_metric_only=True, eval_train_metric=True, num_boost_round=500):
    X_train, y_train, X_test, y_test = dataset
    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train)

    # specify your configurations as a dict
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': metric_list,
        # 'metric': ['auc', 'binary_logloss',],
        'num_leaves': 8,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }

    res_dict = {}
    callbacks = [
        lgb.early_stopping(stopping_rounds=100, first_metric_only=first_metric_only, verbose=True),
        # lgb.record_evaluation(res_dict)
    ]

    # Training settings
    FOLD_NUM = 5
    fold_seed = 71
    folds = StratifiedKFold(n_splits=FOLD_NUM, shuffle=True, random_state=fold_seed)

    # Fitting
    ret = lgb.cv(params=params,
                 train_set=lgb_train,
                 folds=folds,
                 num_boost_round=num_boost_round,
                 verbose_eval=50,
                 callbacks=callbacks,
                 eval_train_metric=eval_train_metric,
                 # return_boosters=True
                 )

    # print(f"best_iteration : {model.best_iteration}")
    df_ret = pd.DataFrame(ret)
    display(df_ret.tail())
    return df_ret


def classification_param():
    two_metrics = list(permutations(['binary_logloss', 'auc', ], 2))
    three_metrics = list(permutations(['binary_logloss', 'auc', "xentropy"], 3))

    param_grid = {
        #        "metric_list" : [two_metrics[0]],
        "metric_list": two_metrics + three_metrics + [
            None, "", "None", "binary_logloss", "auc", "xentropy",  # just one metric
        ],
        # "metric_list" : [""],
        "first_metric_only": [True, False],
        "eval_train_metric": [True, False],
    }
    pg = ParameterGrid(param_grid)
    return pg

def draw_metric_graph(df_ret):
    valid_cols = [c for c in df_ret.columns if c.find("valid") >= 0 and c.find("mean") >= 0]
    train_cols = [c for c in df_ret.columns if c.find("train") >= 0 and c.find("mean") >= 0]

    if len(valid_cols) == 0:
        valid_cols = [c for c in df_ret.columns if c.find("mean") >= 0]

    n_graph = len(valid_cols) + len(train_cols)
    n_cols = 3
    n_rows = n_graph // n_cols + 1

    plt.figure(figsize=(25, 5 * n_rows))
    cnt = 1
    for c in valid_cols:
        plt.subplot(n_rows, n_cols, cnt);
        cnt += 1
        df_ret[c].plot()
        plt.title(f"{c}")
        # plt.show()

    for c in train_cols:
        plt.subplot(n_rows, n_cols, cnt);
        cnt += 1
        df_ret[c].plot()
        plt.title(f"{c}")
        # plt.show()
    plt.tight_layout()
    plt.show()


def test_train(train_func, dataset):
    pg = classification_param()
    num_boost_round = 300
    for p in pg:
        print("=" * 100, flush=True)
        print("=" * 100, flush=True)
        print(p, flush=True)
        try:
            df_ret = train_func(dataset, metric_list=p["metric_list"],
                                          eval_train_metric=p["eval_train_metric"],
                                          first_metric_only=p["first_metric_only"],
                                          num_boost_round=num_boost_round)
            draw_metric_graph(df_ret)
            if df_ret.shape[0] == num_boost_round:
                raise Exception("early_stopping was not applied.")
        except Exception as e:
            traceback.print_exc()

        print("", flush=True)
        # break


X_train, y_train, X_test, y_test = load_data_for_classification()
test_train(classification_train, dataset=(X_train, y_train, X_test, y_test))