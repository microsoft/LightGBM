# coding: utf-8
import glob
import numpy as np

preds = [np.loadtxt(name) for name in glob.glob('*.pred')]
np.testing.assert_array_almost_equal(preds[0], preds[1], decimal=5)
