# coding: utf-8
import glob
import numpy as np

preds = []
for name in glob.glob('*.pred'):
    print('find ' + name + '!')
    preds.append(np.loadtxt(name))

try:
    np.testing.assert_array_equal(preds[0], preds[1])
except AssertionError:
    print('Two arrays are different!')  # ensure using different predictions

np.testing.assert_array_almost_equal(preds[0], preds[1], decimal=5)
