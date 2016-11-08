import sys
import os
import ctypes
import collections

import numpy as np
from scipy import sparse

def LoadDll():
    """Load xgboost Library."""
    lib_path = '../../windows/x64/DLL/lib_lightgbm.dll'
    if len(lib_path) == 0:
        return None
    lib = ctypes.cdll.LoadLibrary(lib_path)
    return lib

LIB = LoadDll()

def c_array(ctype, values):
    return (ctype * len(values))(*values)

def c_str(string):
    return ctypes.c_char_p(string.encode('utf-8'))

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
    csr = sparse.csr_matrix(mat)
    handle = ctypes.c_void_p()
    ref = None
    if reference != None:
        ref = ctypes.byref(reference)

    LIB.LGBM_CreateDatasetFromCSR(c_array(ctypes.c_int, csr.indptr), 2, 
        c_array(ctypes.c_int, csr.indices), 
        csr.data.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p)),
        1, len(csr.indptr), len(csr.data),
        csr.shape[1], ctypes.c_char_p('max_bin=15'), ref, ctypes.byref(handle) )

    num_data = ctypes.c_ulong()
    LIB.LGBM_DatasetGetNumData(handle, ctypes.byref(num_data) )
    num_feature = ctypes.c_ulong()
    LIB.LGBM_DatasetGetNumFeature(handle, ctypes.byref(num_feature) )

    LIB.LGBM_DatasetSetField(handle, c_str('label'), c_array(ctypes.c_float, label), len(label), 0)
    return handle


train = test_load_from_csr('../../examples/binary_classification/binary.train', None)
test = [test_load_from_csr('../../examples/binary_classification/binary.test', train)]
name = [c_str('test')]
booster = ctypes.c_void_p()
LIB.LGBM_BoosterCreate(train, c_array(ctypes.c_void_p, test), c_array(ctypes.c_char_p, name), 
    len(test), "app=binary metric=auc num_leaves=31 verbose=0", ctypes.byref(booster))
is_finished = ctypes.c_int(0)
for i in xrange(100):
    LIB.LGBM_BoosterUpdateOneIter(booster,ctypes.byref(is_finished))
    result = np.array([0.0], dtype=np.float32)
    out_len = ctypes.c_ulong(0)
    LIB.LGBM_BoosterEval(booster, 1, ctypes.byref(out_len), result.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    print '%d Iteration test AUC %f' %(i, result[0])
LIB.LGBM_BoosterSaveModel(booster, -1, c_str('model.txt'))

booster2 = ctypes.c_void_p()
LIB.LGBM_BoosterLoadFromModelfile(c_str('model.txt'), ctypes.byref(booster2))
