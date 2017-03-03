# coding: utf-8
# pylint: disable = C0103

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .basic import LightGBMError
from .compat import range_


class ForestLayer(object):
    """one layer of CascadeForest"""

    def __init__(self, num_forest, num_tree):
        self.num_forest = num_forest
        self.num_tree = num_tree
        self.forests = None

    def fit(self, X, y):
        self.forests = []
        for _ in range_(self.num_forest):
            forest = RandomForestClassifier(self.num_tree)  # to do
            forest.fit(X, y)
            self.forests.append(forest)

    def transfrom(self, X):
        if self.forests is None:
            raise LightGBMError('Need to call fit before transfrom.')
        ret = []
        for forest in self.forests:
            classVector = forest.predict_proba(X)  # to do
            ret.append(classVector)
        return np.concatenate(ret, axis=1)

    def fit_transfrom(self, X, y):
        self.fit(X, y)
        return self.transfrom(X)


class CascadeForest(object):
    """implement of Cascade Forest"""

    def __init__(self, num_forest=4, num_tree=1000):
        self.num_forest = num_forest
        self.num_tree = num_tree
        self.layers = []

    def train(self, X, y):
        classVector = np.array([])
        while True:  # to do: 1. change condition 2. add cv
            X = np.concatenate([X, classVector], axis=1)
            forestLayer = ForestLayer(self.num_forest, self.num_tree)
            classVector = forestLayer.fit_transfrom(X, y)
            self.layers.append(forestLayer)
        return classVector

    def predict(self, X):
        classVector = np.array([])
        for forestLayer in self.layers:
            X = np.concatenate([X, classVector], axis=1)
            classVector = forestLayer.transfrom(X)
        return np.argmax(classVector, axis=1)


class MultiGrainedScanning(object):
    """implement of Multi-Grained Scanning"""

    def __init__(self, num_forest=2, num_tree=30, window_size=10):
        self.forestLayer = ForestLayer(num_forest, num_tree)
        self.window_size = window_size

    def _iter_over_matrix(self, X):
        # to do : support all dims (only support 2d and 3d now)
        # matrix -> list of matrix
        ret = []
        dim = len(X.shape)
        def slide_window(array):
            for i in range_(len(array.shape[1]) - self.window_size + 1):
                ret.append(array[..., i: i + self.window_size])
        if dim is 2:
            slide_window(X)
        elif dim is 3:
            def slide_window_2D(X):
                for i in range_(X.shape[-1]):
                    slide_window(X[..., i])
            slide_window_2D(X)
            np.transpose(X, [0, 2, 1])
            slide_window_2D(X)
        return ret

    def fit(self, X, y, raw_data=True):
        if raw_data:
            X = self._iter_over_matrix(X)
        y = np.tile(y, len(X))
        X = np.concatenate(X)
        self.forestLayer.fit(X, y)

    def transfrom(self, X, raw_data=True):
        if raw_data:
            X = self._iter_over_matrix(X)
        ret = []
        for subX in X:
            ret.append(self.forestLayer.transfrom(subX))
        return np.concatenate(ret, axis=1)

    def fit_transfrom(self, X, y):
        X = self._iter_over_matrix(X)
        self.fit(X, y, False)
        return self.transfrom(X, False)
