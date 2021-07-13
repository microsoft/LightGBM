# coding: utf-8
import ctypes
from os import environ
from pathlib import Path
from platform import system

import numpy as np
from scipy import sparse


def find_lib_path():
    if environ.get('LIGHTGBM_BUILD_DOC', False):
        # we don't need lib_lightgbm while building docs
        return []

    curr_path = Path(__file__).absolute().parent
    dll_path = [curr_path,
                curr_path.parents[1],
                curr_path.parents[1] / 'python-package' / 'lightgbm' / 'compile',
                curr_path.parents[1] / 'python-package' / 'compile',
                curr_path.parents[1] / 'lib']
    if system() in ('Windows', 'Microsoft'):
        dll_path.append(curr_path.parents[1] / 'python-package' / 'compile' / 'Release/')
        dll_path.append(curr_path.parents[1] / 'python-package' / 'compile' / 'windows' / 'x64' / 'DLL')
        dll_path.append(curr_path.parents[1] / 'Release')
        dll_path.append(curr_path.parents[1] / 'windows' / 'x64' / 'DLL')
        dll_path = [p / 'lib_lightgbm.dll' for p in dll_path]
    else:
        dll_path = [p / 'lib_lightgbm.so' for p in dll_path]
    lib_path = [str(p) for p in dll_path if p.is_file()]
    if not lib_path:
        dll_path_joined = '\n'.join(map(str, dll_path))
        raise Exception(f'Cannot find lightgbm library file in following paths:\n{dll_path_joined}')
    return lib_path


def LoadDll():
    lib_path = find_lib_path()
    if len(lib_path) == 0:
        return None
    lib = ctypes.cdll.LoadLibrary(lib_path[0])
    return lib


LIB = LoadDll()

LIB.LGBM_GetLastError.restype = ctypes.c_char_p

dtype_float32 = 0
dtype_float64 = 1
dtype_int32 = 2
dtype_int64 = 3


def c_str(string):
    return ctypes.c_char_p(string.encode('utf-8'))


def load_from_file(filename, reference):
    ref = None
    if reference is not None:
        ref = reference
    handle = ctypes.c_void_p()
    LIB.LGBM_DatasetCreateFromFile(
        c_str(str(filename)),
        c_str('max_bin=15'),
        ref,
        ctypes.byref(handle))
    print(LIB.LGBM_GetLastError())
    num_data = ctypes.c_int(0)
    LIB.LGBM_DatasetGetNumData(handle, ctypes.byref(num_data))
    num_feature = ctypes.c_int(0)
    LIB.LGBM_DatasetGetNumFeature(handle, ctypes.byref(num_feature))
    print(f'#data: {num_data.value} #feature: {num_feature.value}')
    return handle


def save_to_binary(handle, filename):
    LIB.LGBM_DatasetSaveBinary(handle, c_str(filename))


def load_from_csr(filename, reference):
    data = np.loadtxt(str(filename), dtype=np.float64)
    csr = sparse.csr_matrix(data[:, 1:])
    label = data[:, 0].astype(np.float32)
    handle = ctypes.c_void_p()
    ref = None
    if reference is not None:
        ref = reference

    LIB.LGBM_DatasetCreateFromCSR(
        csr.indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int(dtype_int32),
        csr.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        csr.data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(dtype_float64),
        ctypes.c_int64(len(csr.indptr)),
        ctypes.c_int64(len(csr.data)),
        ctypes.c_int64(csr.shape[1]),
        c_str('max_bin=15'),
        ref,
        ctypes.byref(handle))
    num_data = ctypes.c_int(0)
    LIB.LGBM_DatasetGetNumData(handle, ctypes.byref(num_data))
    num_feature = ctypes.c_int(0)
    LIB.LGBM_DatasetGetNumFeature(handle, ctypes.byref(num_feature))
    LIB.LGBM_DatasetSetField(
        handle,
        c_str('label'),
        label.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(len(label)),
        ctypes.c_int(dtype_float32))
    print(f'#data: {num_data.value} #feature: {num_feature.value}')
    return handle


def load_from_csc(filename, reference):
    data = np.loadtxt(str(filename), dtype=np.float64)
    csc = sparse.csc_matrix(data[:, 1:])
    label = data[:, 0].astype(np.float32)
    handle = ctypes.c_void_p()
    ref = None
    if reference is not None:
        ref = reference

    LIB.LGBM_DatasetCreateFromCSC(
        csc.indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int(dtype_int32),
        csc.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        csc.data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(dtype_float64),
        ctypes.c_int64(len(csc.indptr)),
        ctypes.c_int64(len(csc.data)),
        ctypes.c_int64(csc.shape[0]),
        c_str('max_bin=15'),
        ref,
        ctypes.byref(handle))
    num_data = ctypes.c_int(0)
    LIB.LGBM_DatasetGetNumData(handle, ctypes.byref(num_data))
    num_feature = ctypes.c_int(0)
    LIB.LGBM_DatasetGetNumFeature(handle, ctypes.byref(num_feature))
    LIB.LGBM_DatasetSetField(
        handle,
        c_str('label'),
        label.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(len(label)),
        ctypes.c_int(dtype_float32))
    print(f'#data: {num_data.value} #feature: {num_feature.value}')
    return handle


def load_from_mat(filename, reference):
    mat = np.loadtxt(str(filename), dtype=np.float64)
    label = mat[:, 0].astype(np.float32)
    mat = mat[:, 1:]
    data = np.array(mat.reshape(mat.size), dtype=np.float64, copy=False)
    handle = ctypes.c_void_p()
    ref = None
    if reference is not None:
        ref = reference

    LIB.LGBM_DatasetCreateFromMat(
        data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(dtype_float64),
        ctypes.c_int32(mat.shape[0]),
        ctypes.c_int32(mat.shape[1]),
        ctypes.c_int(1),
        c_str('max_bin=15'),
        ref,
        ctypes.byref(handle))
    num_data = ctypes.c_int(0)
    LIB.LGBM_DatasetGetNumData(handle, ctypes.byref(num_data))
    num_feature = ctypes.c_int(0)
    LIB.LGBM_DatasetGetNumFeature(handle, ctypes.byref(num_feature))
    LIB.LGBM_DatasetSetField(
        handle,
        c_str('label'),
        label.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(len(label)),
        ctypes.c_int(dtype_float32))
    print(f'#data: {num_data.value} #feature: {num_feature.value}')
    return handle


def free_dataset(handle):
    LIB.LGBM_DatasetFree(handle)


def test_dataset():
    binary_example_dir = Path(__file__).absolute().parents[2] / 'examples' / 'binary_classification'
    train = load_from_file(binary_example_dir / 'binary.train', None)
    test = load_from_mat(binary_example_dir / 'binary.test', train)
    free_dataset(test)
    test = load_from_csr(binary_example_dir / 'binary.test', train)
    free_dataset(test)
    test = load_from_csc(binary_example_dir / 'binary.test', train)
    free_dataset(test)
    save_to_binary(train, 'train.binary.bin')
    free_dataset(train)
    train = load_from_file('train.binary.bin', None)
    free_dataset(train)


def test_booster():
    binary_example_dir = Path(__file__).absolute().parents[2] / 'examples' / 'binary_classification'
    train = load_from_mat(binary_example_dir / 'binary.train', None)
    test = load_from_mat(binary_example_dir / 'binary.test', train)
    booster = ctypes.c_void_p()
    LIB.LGBM_BoosterCreate(
        train,
        c_str("app=binary metric=auc num_leaves=31 verbose=0"),
        ctypes.byref(booster))
    LIB.LGBM_BoosterAddValidData(booster, test)
    is_finished = ctypes.c_int(0)
    for i in range(1, 51):
        LIB.LGBM_BoosterUpdateOneIter(booster, ctypes.byref(is_finished))
        result = np.array([0.0], dtype=np.float64)
        out_len = ctypes.c_int(0)
        LIB.LGBM_BoosterGetEval(
            booster,
            ctypes.c_int(0),
            ctypes.byref(out_len),
            result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        if i % 10 == 0:
            print(f'{i} iteration test AUC {result[0]:.6f}')
    LIB.LGBM_BoosterSaveModel(
        booster,
        ctypes.c_int(0),
        ctypes.c_int(-1),
        ctypes.c_int(0),
        c_str('model.txt'))
    LIB.LGBM_BoosterFree(booster)
    free_dataset(train)
    free_dataset(test)
    booster2 = ctypes.c_void_p()
    num_total_model = ctypes.c_int(0)
    LIB.LGBM_BoosterCreateFromModelfile(
        c_str('model.txt'),
        ctypes.byref(num_total_model),
        ctypes.byref(booster2))
    data = np.loadtxt(str(binary_example_dir / 'binary.test'), dtype=np.float64)
    mat = data[:, 1:]
    preb = np.empty(mat.shape[0], dtype=np.float64)
    num_preb = ctypes.c_int64(0)
    data = np.array(mat.reshape(mat.size), dtype=np.float64, copy=False)
    LIB.LGBM_BoosterPredictForMat(
        booster2,
        data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(dtype_float64),
        ctypes.c_int32(mat.shape[0]),
        ctypes.c_int32(mat.shape[1]),
        ctypes.c_int(1),
        ctypes.c_int(1),
        ctypes.c_int(0),
        ctypes.c_int(25),
        c_str(''),
        ctypes.byref(num_preb),
        preb.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    LIB.LGBM_BoosterPredictForFile(
        booster2,
        c_str(str(binary_example_dir / 'binary.test')),
        ctypes.c_int(0),
        ctypes.c_int(0),
        ctypes.c_int(0),
        ctypes.c_int(25),
        c_str(''),
        c_str('preb.txt'))
    LIB.LGBM_BoosterPredictForFile(
        booster2,
        c_str(str(binary_example_dir / 'binary.test')),
        ctypes.c_int(0),
        ctypes.c_int(0),
        ctypes.c_int(10),
        ctypes.c_int(25),
        c_str(''),
        c_str('preb.txt'))
    LIB.LGBM_BoosterFree(booster2)
