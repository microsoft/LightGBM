import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest


def simple_pd(X, estimator, feature, values):
    """Calculate simple partial dependency."""
    Xc = X.copy()
    yps = np.zeros_like(values)
    for i, x in enumerate(values):
        Xc[feature] = x
        yps[i] = np.mean(estimator.predict(Xc.values))
    return values, yps


def simple_pds(df, estimator, features):
    """Calculate simple partial dependency for all features."""
    pds = {}
    for feat in features:
        values, yps = simple_pd(
            df[features], estimator, feat, np.sort(df[feat].unique())
        )
        pds[feat] = pd.DataFrame(data={feat: values, "y": yps})
    return pds


@pytest.fixture
def make_data():
    """Make toy data."""
    np.random.seed(1)
    n = 10000
    d = 3
    X = np.random.normal(size=(n, d))
    # round to speed things up
    X = np.round(X)
    eps = np.random.normal() * 0.1
    y = -1 * X[:, 0] + 3 * X[:, 1] + X[:, 0] * X[:, 1] + np.cos(X[:, 2]) + eps
    df = pd.DataFrame(data={"y": y, "x0": X[:, 0], "x1": X[:, 1], "x2": X[:, 2]})
    features = ["x0", "x1", "x2"]
    outcome = "y"

    return df, features, outcome


def find_interactions(gbm, feature_sets):
    """Find interactions in tree.

    Parameters
    ---------
    gbm: booster
    feature_sets: list of list
        set of features across which to check for interactions

    Returns
    -------
    tree_features: pandas.DataFrame
        boolean flag for every tree of has interaction across feature sets.
    """
    df_trees = gbm.trees_to_dataframe()
    tree_features = (
        df_trees.groupby("tree_index")
        .apply(lambda x: set(x["split_feature"]) - set([None]))
        .reset_index()
        .rename(columns={0: "features"})
    )

    def has_interaction(tree):
        n = 0
        for fs in feature_sets:
            if len(tree["features"].intersection(fs)) > 0:
                n += 1
        if n > 1:
            return True
        else:
            return False

    tree_features["has_interaction"] = tree_features.apply(has_interaction, axis=1)

    return tree_features


@pytest.fixture
def get_boosting_params():
    boosting_params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "num_leaves": 5,
        "learning_rate": 0.1,
        "num_boost_round": 100,
        # disallow all interactions
        "interaction_constraints": [[0], [1], [2]],
        "monotone_constraints": [1, 1, 0],
        "monotone_constraints_method": "basic",
    }
    return boosting_params


@pytest.mark.parametrize("monotone_constraints_method", ["basic", "intermediate", "advanced"])
def test_interaction_constraints(make_data, get_boosting_params, monotone_constraints_method):

    df, features, outcome = make_data
    data = lgb.Dataset(df[features], df[outcome])

    boosting_params = get_boosting_params
    boosting_params.update({"monotone_constraints_method": monotone_constraints_method})
    gbm = lgb.train(boosting_params, data)

    feature_sets = [[0], [1], [2]]
    feature_sets = [[f"x{f}" for f in fs] for fs in feature_sets]
    tree_features = find_interactions(gbm, feature_sets)

    # Should not find any co-occurances in a given tree, since above we're disallowing all interactions.
    assert not tree_features["has_interaction"].any()

    # Check monotonicity
    pds = simple_pds(df, gbm, features)
    cnt = 0
    for feat, df_pd in pds.items():
        df_pd = df_pd.sort_values(by=feat, ascending=True)
        y_pred_diff = df_pd["y"].diff().values[1:]
        if boosting_params["monotone_constraints"][cnt] == 1:
            assert (y_pred_diff >= 0).all()
        elif boosting_params["monotone_constraints"][cnt] == -1:
            assert (y_pred_diff <= 0).all()
        else:
            pass
        cnt += 1
