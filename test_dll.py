from __future__ import absolute_import

import sys
import os
import ctypes
import collections
import re

import numpy as np
from scipy import sparse

def _load_lib():
    """Load xgboost Library."""
    lib_path = './windows/x64/DLL/lib_lightgbm.dll'
    if len(lib_path) == 0:
        return None
    lib = ctypes.cdll.LoadLibrary(lib_path)
    return lib


LIB = _load_lib()
LIB.LGBM_GetLastError.restype = ctypes.c_char_p
def test_load_from_file():
    handle = ctypes.c_void_p()
    LIB.LGBM_CreateDatasetFromFile(ctypes.c_char_p('./examples/binary_classification/binary.train'),
    ctypes.c_char_p('max_bin=15'), ctypes.c_void_p(None), ctypes.byref(handle))
    num_data = ctypes.c_ulong()
    LIB.LGBM_DatasetGetNumData(handle, ctypes.byref(num_data) )
    print num_data
    num_feature = ctypes.c_ulong()
    LIB.LGBM_DatasetGetNumFeature(handle, ctypes.byref(num_feature) )
    return handle

def c_array(ctype, values):
    """Convert a python string to c array."""
    return (ctype * len(values))(*values)

def c_str(string):
    """Convert a python string to cstring."""
    return ctypes.c_char_p(string.encode('utf-8'))

def test_load_from_matric():
    data = []

    inp = open('./examples/binary_classification/binary.train', 'r')
    for line in inp.readlines():
        data.append( [float(x) for x in line.split('\t')[1:]] )
    inp.close()
    mat = np.array(data)
    print mat.shape
    data = np.array(mat.reshape(mat.size), copy=False)
    handle = ctypes.c_void_p()
    LIB.LGBM_CreateDatasetFromMat(data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1, 
        mat.shape[0], 
        mat.shape[1], 1, 
    ctypes.c_char_p('max_bin=15 is_sparse=false'), None, ctypes.byref(handle) )
    LIB.LGBM_DatasetFree(ctypes.byref(handle))
    # num_data = ctypes.c_ulong()
    # LIB.LGBM_DatasetGetNumData(handle, ctypes.byref(num_data) )
    # print num_data
    # num_feature = ctypes.c_ulong()
    # LIB.LGBM_DatasetGetNumFeature(handle, ctypes.byref(num_feature) )
    # print num_feature
    return handle

def test_load_from_csr(filename, reference):
    data = []
    label = []
    inp = open(filename, 'r')
    for line in inp.readlines():
        data.append( [float(x) for x in line.split('\t')[1:]] )
        label.append( float(line.split('\t')[0]) )
    inp.close()
    mat = np.array(data)
    label = np.array(label, dtype=np.float32)
    print mat.shape
    csr = sparse.csr_matrix(mat)
    handle = ctypes.c_void_p()
    ref = None
    if reference != None:
        ref = ctypes.byref(reference)
    LIB.LGBM_CreateDatasetFromCSR(c_array(ctypes.c_int, csr.indptr), 
        c_array(ctypes.c_int, csr.indices), 
        csr.data.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p)),
        1, len(csr.indptr), len(csr.data),
        csr.shape[1], ctypes.c_char_p('max_bin=15'), ref, ctypes.byref(handle) )
    num_data = ctypes.c_ulong()
    LIB.LGBM_DatasetGetNumData(handle, ctypes.byref(num_data) )
    print num_data
    num_feature = ctypes.c_ulong()
    LIB.LGBM_DatasetGetNumFeature(handle, ctypes.byref(num_feature) )
    LIB.LGBM_DatasetSetField(handle, c_str('label'), c_array(ctypes.c_float, label), len(label), 0)
    return handle


train = test_load_from_csr('./examples/binary_classification/binary.train', None)
test = [test_load_from_csr('./examples/binary_classification/binary.test', train)]
name = [c_str('test')]
booster = ctypes.c_void_p()
LIB.LGBM_BoosterCreate(train, c_array(ctypes.c_void_p, test), c_array(ctypes.c_char_p, name), 1, "app=binary metric=auc num_leaves=31", ctypes.byref(booster))
is_finished = ctypes.c_int(0)
for i in xrange(100):
    LIB.LGBM_BoosterUpdateOneIter(booster,ctypes.byref(is_finished))
    result = np.array([0.0], dtype=np.float32)
    out_len = ctypes.c_ulong(0)
    LIB.LGBM_BoosterEval(booster, 0, ctypes.byref(out_len), result.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    print result
LIB.LGBM_BoosterSaveModel(booster, -1, c_str('model.txt'))
booster2 = ctypes.c_void_p()

LIB.LGBM_BoosterLoadFromModelfile(c_str('model.txt'), ctypes.byref(booster2))

print type(len([0,0]))