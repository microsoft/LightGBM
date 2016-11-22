"""Wrapper c_api of LightGBM"""
from __future__ import absolute_import

import sys
import os
import ctypes
import collections
import re

import numpy as np
import scipy.sparse


IS_PY3 = (sys.version_info[0] == 3)


def find_lib_path():
    """Find the path to LightGBM library files.
    Returns
    -------
    lib_path: list(string)
       List of all found library path to LightGBM
    """
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    dll_path = [curr_path, os.path.join(curr_path, '../../lib/'),
                os.path.join(curr_path, './lib/'),
                os.path.join(sys.prefix, 'lightgbm')]
    if os.name == 'nt':
        dll_path.append(os.path.join(curr_path, '../../windows/x64/Dll/'))
        dll_path.append(os.path.join(curr_path, './windows/x64/Dll/'))
        dll_path = [os.path.join(p, 'lib_lightgbm.dll') for p in dll_path]
    else:
        dll_path = [os.path.join(p, 'lib_lightgbm.so') for p in dll_path]
    lib_path = [p for p in dll_path if os.path.exists(p) and os.path.isfile(p)]
    if not lib_path:
        raise Exception('Cannot find lightgbm Library')
    return lib_path

def _load_lib():
    """Load LightGBM Library."""
    lib_path = find_lib_path()
    if len(lib_path) == 0:
        return None
    lib = ctypes.cdll.LoadLibrary(lib_path[0])
    lib.LGBM_GetLastError.restype = ctypes.c_char_p
    return lib

_LIB = _load_lib()

class LightGBMError(Exception):
    """Error throwed by LightGBM"""
    pass

def _safe_call(ret):
    """Check the return value of C API call
    Parameters
    ----------
    ret : int
        return value from API calls
    """
    if ret != 0:
        raise LightGBMError(_LIB.LGBM_GetLastError())

def is_str(s):
    if IS_PY3:
        return isinstance(s, str)
    else:
        return isinstance(s, basestring)

def is_numpy_object(data):
    return type(data).__module__ == np.__name__

def is_numpy_1d_array(data):
    if isinstance(data, np.ndarray) and len(data.shape) == 1:
        return True
    else:
        return False

def list_to_1d_numpy(data, dtype):
    if is_numpy_1d_array(data):
        return data
    elif isinstance(data, list):
        return np.array(data, dtype=dtype, copy=False)
    else:
        raise TypeError("Unknow type({})".format(type(data).__name__))

def cfloat32_array_to_numpy(cptr, length):
    """Convert a ctypes float pointer array to a numpy array.
    """
    if isinstance(cptr, ctypes.POINTER(ctypes.c_float)):
        res = np.fromiter(cptr, dtype=np.float32, count=length)
        return res
    else:
        raise RuntimeError('expected float pointer')

def cint32_array_to_numpy(cptr, length):
    """Convert a ctypes float pointer array to a numpy array.
    """
    if isinstance(cptr, ctypes.POINTER(ctypes.c_int32)):
        res = np.fromiter(cptr, dtype=np.int32, count=length)
        return res
    else:
        raise RuntimeError('expected int pointer')

def c_str(string):
    """Convert a python string to cstring."""
    return ctypes.c_char_p(string.encode('utf-8'))

def c_array(ctype, values):
    """Convert a python array to c array."""
    return (ctype * len(values))(*values)

"""marco definition of data type in c_api of LightGBM"""
C_API_DTYPE_FLOAT32 =0
C_API_DTYPE_FLOAT64 =1
C_API_DTYPE_INT32   =2
C_API_DTYPE_INT64   =3
"""Matric is row major in python"""
C_API_IS_ROW_MAJOR  =1

def c_float_array(data):
    """Convert numpy array / list to c float array."""
    if isinstance(data, list):
        data = np.array(data, copy=False)
    if is_numpy_1d_array(data):
        if data.dtype == np.float32:
            ptr_data = c_array(ctypes.c_float, data)
            type_data = C_API_DTYPE_FLOAT32
        elif data.dtype == np.float64:
            ptr_data = c_array(ctypes.c_double, data)
            type_data = C_API_DTYPE_FLOAT64
        else:
            raise TypeError("expected np.float32 or np.float64, met type({})".format(data.dtype))
    else:
        raise TypeError("Unknow type({})".format(type(data).__name__))
    return (ptr_data, type_data)

def c_int_array(data):
    """Convert numpy array to c int array."""
    if isinstance(data, list):
        data = np.array(data, copy=False)
    if is_numpy_1d_array(data):
        if data.dtype == np.int32:
            ptr_data = c_array(ctypes.c_int32, data)
            type_data = C_API_DTYPE_INT32
        elif data.dtype == np.int64:
            ptr_data = c_array(ctypes.c_int64, data)
            type_data = C_API_DTYPE_INT64
        else:
            raise TypeError("expected np.int32 or np.int64, met type({})".format(data.dtype))
    else:
        raise TypeError("Unknow type({})".format(type(data).__name__))
    return (ptr_data, type_data)

class Dataset(object):
    """Dataset used in LightGBM.

    Dataset is a internal data structure that used by LightGBM
    You can construct Dataset from numpy.arrays
    """

    _feature_names = None

    def __init__(self, data, max_bin=255, reference=None,
        label=None, weight=None, group_id=None, 
        silent=False, feature_names=None, 
        other_args=None):
        """
        Dataset used in LightGBM.

        Parameters
        ----------
        data : string/numpy array/scipy.sparse
            Data source of Dataset.
            When data is string type, it represents the path of txt file,
        max_bin : int, required
            max number of discrete bin for features 
        reference : Other Dataset, optional
            If this dataset validation, need to use training data as reference
        label : list or numpy 1-D array, optional
            Label of the training data.
        weight : list or numpy 1-D array , optional
            Weight for each instance.
        group_id : list or numpy 1-D array , optional
            group/query id for each instance. Note: if having group/query id, data should group by this id
        silent : boolean, optional
            Whether print messages during construction
        feature_names : list, optional
            Set names for features.
        other_args: list, optional
            other parameters, format: ['key1=val1','key2=val2']
        """

        if data is None:
            self.handle = None
            return
        """process for args"""
        pass_args = ["max_bin={}".format(max_bin)]
        if silent:
            pass_args.append("verbose=0")
        if other_args:
            pass_args += other_args
        pass_args_str = ' '.join(pass_args)
        """process for reference dataset"""
        ref_dataset = None
        if isinstance(reference, Dataset):
            ref_dataset = ctypes.byref(reference.handle)
        elif reference is not None:
            raise TypeError('Reference dataset should be None or dataset instance')
        """start construct data"""
        if is_str(data):
            self.handle = ctypes.c_void_p()
            _safe_call(_LIB.LGBM_CreateDatasetFromFile(
                c_str(data), 
                c_str(pass_args_str), 
                ref_dataset,
                ctypes.byref(self.handle)))
        elif isinstance(data, scipy.sparse.csr_matrix):
            self._init_from_csr(data, pass_args_str, ref_dataset)
        elif isinstance(data, scipy.sparse.csc_matrix):
            self._init_from_csc(data, pass_args_str, ref_dataset)
        elif isinstance(data, np.ndarray):
            self._init_from_npy2d(data, pass_args_str, ref_dataset)
        else:
            try:
                csr = scipy.sparse.csr_matrix(data)
                self._init_from_csr(csr)
            except:
                raise TypeError('can not initialize Dataset from {}'.format(type(data).__name__))
        if label is not None:
            self.set_label(label)
        if weight is not None:
            self.set_weight(weight)
        if group_id is not None:
            self.set_group_id(group_id)
        self.feature_names = feature_names

    def _init_from_csr(self, csr, pass_args_str, ref_dataset):
        """
        Initialize data from a CSR matrix.
        """
        if len(csr.indices) != len(csr.data):
            raise ValueError('length mismatch: {} vs {}'.format(len(csr.indices), len(csr.data)))
        self.handle = ctypes.c_void_p()

        ptr_indptr, type_ptr_indptr = c_int_array(csr.indptr)
        ptr_data, type_ptr_data = c_float_array(csr.data)

        _safe_call(_LIB.LGBM_CreateDatasetFromCSR(
            ptr_indptr, 
            type_ptr_indptr, 
            c_array(ctypes.c_int32, csr.indices), 
            ptr_data,
            type_ptr_data, 
            len(csr.indptr), 
            len(csr.data),
            csr.shape[1], 
            c_str(pass_args_str), 
            ref_dataset, 
            ctypes.byref(self.handle)))

    def _init_from_csc(self, csr, pass_args_str, ref_dataset):
        """
        Initialize data from a CSC matrix.
        """
        if len(csc.indices) != len(csc.data):
            raise ValueError('length mismatch: {} vs {}'.format(len(csc.indices), len(csc.data)))
        self.handle = ctypes.c_void_p()

        ptr_indptr, type_ptr_indptr = c_int_array(csc.indptr)
        ptr_data, type_ptr_data = c_float_array(csc.data)

        _safe_call(_LIB.LGBM_CreateDatasetFromCSC(
            ptr_indptr, 
            type_ptr_indptr, 
            c_array(ctypes.c_int32, csc.indices), 
            ptr_data,
            type_ptr_data, 
            len(csc.indptr), 
            len(csc.data),
            csc.shape[0], 
            c_str(pass_args_str), 
            ref_dataset, 
            ctypes.byref(self.handle)))

    def _init_from_npy2d(self, mat, pass_args_str, ref_dataset):
        """
        Initialize data from a 2-D numpy matrix.
        """
        if len(mat.shape) != 2:
            raise ValueError('Input numpy.ndarray must be 2 dimensional')

        self.handle = ctypes.c_void_p()
        if mat.dtype == np.float32 or mat.dtype == np.float64:
            data = np.array(mat.reshape(mat.size), dtype=mat.dtype, copy=False)
        else:
            """change non-float data to float data, need to copy"""
            data = np.array(mat.reshape(mat.size), dtype=np.float32)

        ptr_data, type_ptr_data = c_float_array(data)
        _safe_call(LIB.LGBM_CreateDatasetFromMat(
            ptr_data, 
            type_ptr_data,
            mat.shape[0],
            mat.shape[1],
            C_API_IS_ROW_MAJOR,
            c_str(pass_args_str), 
            ref_dataset, 
            ctypes.byref(self.handle)))

    def __del__(self):
        _safe_call(_LIB.LGBM_DatasetFree(self.handle))

    def get_field(self, field_name):
        """Get property from the Dataset.

        Parameters
        ----------
        field_name: str
            The field name of the information

        Returns
        -------
        info : array
            a numpy array of information of the data
        """
        out_len = ctypes.c_int32()
        out_type = ctypes.c_int32()
        ret = ctypes.POINTER(ctypes.c_void_p)()
        _safe_call(_LIB.LGBM_DatasetGetField(
            self.handle,
            c_str(field_name),
            ctypes.byref(out_len),
            ctypes.byref(ret),
            ctypes.byref(out_type)))
        if out_type.value == C_API_DTYPE_INT32:
            return cint32_array_to_numpy(ctypes.cast(ret, ctypes.POINTER(c_int32), out_len.value))
        elif out_type.value == C_API_DTYPE_FLOAT32:
            return cfloat32_array_to_numpy(ctypes.cast(ret, ctypes.POINTER(c_float), out_len.value))
        else:
            raise TypeError("unknow type")

    def set_field(self, field_name, data):
        """Set property into the Dataset.

        Parameters
        ----------
        field_name: str
            The field name of the information

        data: numpy array or list
            The array ofdata to be set
        """
        if not is_numpy_1d_array(data):
            raise TypeError("Unknow type({})".format(type(data).__name__))
        if data.dtype == np.float32:
            ptr_data = c_array(ctypes.c_float, data)
            type_data = C_API_DTYPE_FLOAT32
        elif data.dtype == np.int32:
            ptr_data = c_array(ctypes.c_int32, data)
            type_data = C_API_DTYPE_INT32
        else:
            raise TypeError("excepted np.float32 or np.int32, met type({})".format(data.dtype))
        _safe_call(_LIB.LGBM_DatasetSetField(
            self.handle,
            c_str(field_name),
            ptr_data,
            len(data),
            type_data))


    def save_binary(self, filename):
        """Save Dataset to binary file

        Parameters
        ----------
        filename : string
            Name of the output file.
        """
        _safe_call(_LIB.LGBM_DatasetSaveBinary(
            self.handle,
            c_str(filename)))

    def set_label(self, label):
        """Set label of Dataset

        Parameters
        ----------
        label: array like
            The label information to be set into Dataset
        """
        label = list_to_1d_numpy(label, np.float32)
        if label.dtype != np.float32:
            label = label.astype(np.float32, copy=False)
        self.set_field('label', label)

    def set_weight(self, weight):
        """ Set weight of each instance.

        Parameters
        ----------
        weight : array like
            Weight for each data point
        """
        weight = list_to_1d_numpy(weight, np.float32)
        if weight.dtype != np.float32:
            weight = weight.astype(np.float32, copy=False)
        self.set_field('weight', weight)

    def set_init_score(self, score):
        """ Set init score of booster to start from.
        Parameters
        ----------
        score: array like

        """
        score = list_to_1d_numpy(score, np.float32)
        if score.dtype != np.float32:
            score = score.astype(np.float32, copy=False)
        self.set_field('init_score', score)

    def set_group(self, group):
        """Set group size of Dataset (used for ranking).

        Parameters
        ----------
        group : array like
            Group size of each group
        """
        group = list_to_1d_numpy(group, np.int32)
        if group.dtype != np.int32:
            group = group.astype(np.int32, copy=False)
        self.set_field('group', group)

    def set_group_id(self, group_id):

        """Set group_id of Dataset (used for ranking).

        Parameters
        ----------
        group : array like
            group_id of Dataset (used for ranking).
        """
        group_id = list_to_1d_numpy(group_id, np.int32)
        if group_id.dtype != np.int32:
            group_id = group_id.astype(np.int32, copy=False)
        self.set_field('group_id', group_id)

    def get_label(self):
        """Get the label of the Dataset.

        Returns
        -------
        label : array
        """
        return self.get_field('label')

    def get_weight(self):
        """Get the weight of the Dataset.

        Returns
        -------
        weight : array
        """
        return self.get_field('weight')

    def get_init_score(self):
        """Get the initial score of the Dataset.

        Returns
        -------
        init_score : array
        """
        return self.get_field('init_score')

    def num_data(self):
        """Get the number of rows in the Dataset.

        Returns
        -------
        number of rows : int
        """
        ret = ctypes.c_int64()
        _safe_call(_LIB.LGBM_DatasetGetNumData(self.handle,
                                         ctypes.byref(ret)))
        return ret.value

    def num_feature(self):
        """Get the number of columns (features) in the Dataset.

        Returns
        -------
        number of columns : int
        """
        ret = ctypes.c_int64()
        _safe_call(_LIB.LGBM_DatasetGetNumFeature(self.handle,
                                         ctypes.byref(ret)))
        return ret.value

    @property
    def feature_names(self):
        """Get feature names (column labels).

        Returns
        -------
        feature_names : list
        """
        if self._feature_names is None:
            self._feature_names = ['Column_{0}'.format(i) for i in range(self.num_col())]
        return self._feature_names

    @feature_names.setter
    def feature_names(self, feature_names):
        """Set feature names (column labels).

        Parameters
        ----------
        feature_names : list
            Labels for features
        """
        if feature_names is not None:
            # validate feature name
            if not isinstance(feature_names, list):
                feature_names = list(feature_names)
            if len(feature_names) != len(set(feature_names)):
                raise ValueError('feature_names must be unique')
            if len(feature_names) != self.num_col():
                msg = 'feature_names must have the same length as data'
                raise ValueError(msg)
            # prohibit to use symbols may affect to parse. e.g. []<
            if not all(isinstance(f, STRING_TYPES) and
                       not any(x in f for x in set(('[', ']', '<')))
                       for f in feature_names):
                raise ValueError('feature_names may not contain [, ] or <')
            self._feature_names = feature_names
        else:
            self._feature_names = None

