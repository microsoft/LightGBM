# coding: utf-8
# pylint: disable = invalid-name, C0111, C0301, R0912, R0913, R0914, W0105
# pylint: disable = E1101
"""Wrapper c_api of LightGBM"""
from __future__ import absolute_import

import sys
import ctypes
import tempfile
import json

import numpy as np
import scipy.sparse

from .libpath import find_lib_path

# pandas
try:
    from pandas import Series, DataFrame
    IS_PANDAS_INSTALLED = True
except ImportError:
    class Series(object):
        pass
    class DataFrame(object):
        pass
    IS_PANDAS_INSTALLED = False

IS_PY3 = (sys.version_info[0] == 3)

def _load_lib():
    """Load LightGBM Library."""
    lib_path = find_lib_path()
    if len(lib_path) == 0:
        raise Exception("cannot find LightGBM library")
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
    return isinstance(data, np.ndarray) and len(data.shape) == 1

def is_1d_list(data):
    return isinstance(data, list) and \
        (not data or isinstance(data[0], (int, float, bool)))

def list_to_1d_numpy(data, dtype):
    if is_numpy_1d_array(data):
        if data.dtype == dtype:
            return data
        else:
            return data.astype(dtype=dtype, copy=False)
    elif is_1d_list(data):
        return np.array(data, dtype=dtype, copy=False)
    elif IS_PANDAS_INSTALLED and isinstance(data, Series):
        return data.astype(dtype).values
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

def param_dict_to_str(data):
    if not data:
        return ""
    pairs = []
    for key, val in data.items():
        if is_str(val) or isinstance(val, (int, float, bool)):
            pairs.append(str(key)+'='+str(val))
        elif isinstance(val, (list, tuple, set)):
            pairs.append(str(key)+'='+','.join(map(str, val)))
        else:
            raise TypeError('unknow type of parameter:%s , got:%s'
                            % (key, type(val).__name__))
    return ' '.join(pairs)

"""marco definition of data type in c_api of LightGBM"""
C_API_DTYPE_FLOAT32 = 0
C_API_DTYPE_FLOAT64 = 1
C_API_DTYPE_INT32 = 2
C_API_DTYPE_INT64 = 3
"""Matric is row major in python"""
C_API_IS_ROW_MAJOR = 1

C_API_PREDICT_NORMAL = 0
C_API_PREDICT_RAW_SCORE = 1
C_API_PREDICT_LEAF_INDEX = 2

FIELD_TYPE_MAPPER = {"label": C_API_DTYPE_FLOAT32,
                     "weight": C_API_DTYPE_FLOAT32,
                     "init_score": C_API_DTYPE_FLOAT32,
                     "group": C_API_DTYPE_INT32}

def c_float_array(data):
    """Convert numpy array / list to c float array."""
    if is_1d_list(data):
        data = np.array(data, copy=False)
    if is_numpy_1d_array(data):
        if data.dtype == np.float32:
            ptr_data = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            type_data = C_API_DTYPE_FLOAT32
        elif data.dtype == np.float64:
            ptr_data = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            type_data = C_API_DTYPE_FLOAT64
        else:
            raise TypeError("expected np.float32 or np.float64, met type({})"
                            .format(data.dtype))
    else:
        raise TypeError("Unknow type({})".format(type(data).__name__))
    return (ptr_data, type_data)

def c_int_array(data):
    """Convert numpy array to c int array."""
    if is_1d_list(data):
        data = np.array(data, copy=False)
    if is_numpy_1d_array(data):
        if data.dtype == np.int32:
            ptr_data = data.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            type_data = C_API_DTYPE_INT32
        elif data.dtype == np.int64:
            ptr_data = data.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
            type_data = C_API_DTYPE_INT64
        else:
            raise TypeError("expected np.int32 or np.int64, met type({})"
                            .format(data.dtype))
    else:
        raise TypeError("Unknow type({})".format(type(data).__name__))
    return (ptr_data, type_data)

class Predictor(object):
    """"A Predictor of LightGBM.
    """
    def __init__(self, model_file=None, booster_handle=None, is_manage_handle=True):
        """Initialize the Predictor.

        Parameters
        ----------
        model_file : string
            Path to the model file.
        """
        self.handle = ctypes.c_void_p()
        self.__is_manage_handle = True
        if model_file is not None:
            """Prediction task"""
            out_num_iterations = ctypes.c_int64(0)
            _safe_call(_LIB.LGBM_BoosterCreateFromModelfile(
                c_str(model_file),
                ctypes.byref(out_num_iterations),
                ctypes.byref(self.handle)))
            out_num_class = ctypes.c_int64(0)
            _safe_call(_LIB.LGBM_BoosterGetNumClasses(
                self.handle,
                ctypes.byref(out_num_class)))
            self.num_class = out_num_class.value
            self.num_total_iteration = out_num_iterations.value
        elif booster_handle is not None:
            self.__is_manage_handle = is_manage_handle
            self.handle = booster_handle
            out_num_class = ctypes.c_int64(0)
            _safe_call(_LIB.LGBM_BoosterGetNumClasses(
                self.handle,
                ctypes.byref(out_num_class)))
            self.num_class = out_num_class.value
            out_num_iterations = ctypes.c_int64(0)
            _safe_call(_LIB.LGBM_BoosterGetCurrentIteration(
                self.handle,
                ctypes.byref(out_num_iterations)))
            self.num_total_iteration = out_num_iterations.value
        else:
            raise TypeError('Need Model file to create a booster')

    def __del__(self):
        if self.__is_manage_handle:
            _safe_call(_LIB.LGBM_BoosterFree(self.handle))


    def predict(self, data, num_iteration=-1,
                raw_score=False, pred_leaf=False, data_has_header=False,
                is_reshape=True):
        """
        Predict logic

        Parameters
        ----------
        data : string/numpy array/scipy.sparse
            Data source for prediction
            When data type is string, it represents the path of txt file
        num_iteration : int
            Used iteration for prediction
        raw_score : bool
            True for predict raw score
        pred_leaf : bool
            True for predict leaf index
        data_has_header : bool
            Used for txt data
        is_reshape : bool
            Reshape to (nrow, ncol) if true

        Returns
        -------
        Prediction result
        """
        if isinstance(data, (_InnerDataset, Dataset)):
            raise TypeError("cannot use Dataset instance for prediction, please use raw data instead")
        predict_type = C_API_PREDICT_NORMAL
        if raw_score:
            predict_type = C_API_PREDICT_RAW_SCORE
        if pred_leaf:
            predict_type = C_API_PREDICT_LEAF_INDEX
        int_data_has_header = 1 if data_has_header else 0
        if num_iteration > self.num_total_iteration:
            num_iteration = self.num_total_iteration
        if is_str(data):
            tmp_pred_fname = tempfile.NamedTemporaryFile(prefix="lightgbm_tmp_pred_").name
            _safe_call(_LIB.LGBM_BoosterPredictForFile(
                self.handle,
                c_str(data),
                int_data_has_header,
                predict_type,
                num_iteration,
                c_str(tmp_pred_fname)))
            with open(tmp_pred_fname, "r") as tmp_file:
                lines = tmp_file.readlines()
                nrow = len(lines)
                preds = [float(token) for line in lines for token in line.split('\t')]
                preds = np.array(preds, dtype=np.float32, copy=False)
        elif isinstance(data, scipy.sparse.csr_matrix):
            preds, nrow = self.__pred_for_csr(data, num_iteration,
                                              predict_type)
        elif isinstance(data, np.ndarray):
            preds, nrow = self.__pred_for_np2d(data, num_iteration,
                                               predict_type)
        elif IS_PANDAS_INSTALLED and isinstance(data, DataFrame):
            preds, nrow = self.__pred_for_np2d(data.values, num_iteration,
                                               predict_type)
        else:
            try:
                csr = scipy.sparse.csr_matrix(data)
                preds, nrow = self.__pred_for_csr(csr, num_iteration,
                                                  predict_type)
            except:
                raise TypeError('can not predict data for type {}'.
                                format(type(data).__name__))
        if pred_leaf:
            preds = preds.astype(np.int32)
        if is_reshape and preds.size != nrow:
            if preds.size % nrow == 0:
                preds = preds.reshape(nrow, -1)
            else:
                raise ValueError('length of predict result (%d) cannot be divide nrow (%d)'
                                 % (preds.size, nrow))
        return preds

    def __get_num_preds(self, num_iteration, nrow, predict_type):
        n_preds = self.num_class * nrow
        if predict_type == C_API_PREDICT_LEAF_INDEX:
            if num_iteration > 0:
                n_preds *= min(num_iteration, self.num_total_iteration)
            else:
                n_preds *= self.num_total_iteration
        return n_preds

    def __pred_for_np2d(self, mat, num_iteration, predict_type):
        """
        Predict for a 2-D numpy matrix.
        """
        if len(mat.shape) != 2:
            raise ValueError('Input numpy.ndarray must be 2 dimensional')

        if mat.dtype == np.float32 or mat.dtype == np.float64:
            data = np.array(mat.reshape(mat.size), dtype=mat.dtype, copy=False)
        else:
            """change non-float data to float data, need to copy"""
            data = np.array(mat.reshape(mat.size), dtype=np.float32)
        ptr_data, type_ptr_data = c_float_array(data)
        n_preds = self.__get_num_preds(num_iteration, mat.shape[0],
                                       predict_type)
        preds = np.zeros(n_preds, dtype=np.float32)
        out_num_preds = ctypes.c_int64(0)
        _safe_call(_LIB.LGBM_BoosterPredictForMat(
            self.handle,
            ptr_data,
            type_ptr_data,
            mat.shape[0],
            mat.shape[1],
            C_API_IS_ROW_MAJOR,
            predict_type,
            num_iteration,
            ctypes.byref(out_num_preds),
            preds.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            ))
        if n_preds != out_num_preds.value:
            raise ValueError("incorrect number for predict result")
        return preds, mat.shape[0]

    def __pred_for_csr(self, csr, num_iteration, predict_type):
        """
        Predict for a csr data
        """
        nrow = len(csr.indptr) - 1
        n_preds = self.__get_num_preds(num_iteration, nrow, predict_type)
        preds = np.zeros(n_preds, dtype=np.float32)
        out_num_preds = ctypes.c_int64(0)

        ptr_indptr, type_ptr_indptr = c_int_array(csr.indptr)
        ptr_data, type_ptr_data = c_float_array(csr.data)

        _safe_call(_LIB.LGBM_BoosterPredictForCSR(
            self.handle,
            ptr_indptr,
            type_ptr_indptr,
            csr.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ptr_data,
            type_ptr_data,
            len(csr.indptr),
            len(csr.data),
            csr.shape[1],
            predict_type,
            num_iteration,
            ctypes.byref(out_num_preds),
            preds.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            ))
        if n_preds != out_num_preds.value:
            raise ValueError("incorrect number for predict result")
        return preds, nrow

PANDAS_DTYPE_MAPPER = {'int8': 'int', 'int16': 'int', 'int32': 'int',
                       'int64': 'int', 'uint8': 'int', 'uint16': 'int',
                       'uint32': 'int', 'uint64': 'int', 'float16': 'float',
                       'float32': 'float', 'float64': 'float', 'bool': 'int'}

def _data_from_pandas(data):
    if isinstance(data, DataFrame):
        data_dtypes = data.dtypes
        if not all(dtype.name in PANDAS_DTYPE_MAPPER for dtype in data_dtypes):
            bad_fields = [data.columns[i] for i, dtype in
                          enumerate(data_dtypes) if dtype.name not in PANDAS_DTYPE_MAPPER]

            msg = """DataFrame.dtypes for data must be int, float or bool. Did not expect the data types in fields """
            raise ValueError(msg + ', '.join(bad_fields))
        data = data.values.astype('float')
    return data

def _label_from_pandas(label):
    if isinstance(label, DataFrame):
        if len(label.columns) > 1:
            raise ValueError('DataFrame for label cannot have multiple columns')
        label_dtypes = label.dtypes
        if not all(dtype.name in PANDAS_DTYPE_MAPPER for dtype in label_dtypes):
            raise ValueError('DataFrame.dtypes for label must be int, float or bool')
        label = label.values.astype('float')
    return label

class _InnerDataset(object):
    """_InnerDataset used in LightGBM.

    _InnerDataset is a internal data structure that used by LightGBM
    """

    def __init__(self, data, label=None, max_bin=255, reference=None,
                 weight=None, group=None, predictor=None,
                 silent=False, feature_name=None,
                 categorical_feature=None, params=None):
        """
        _InnerDataset used in LightGBM.

        Parameters
        ----------
        data : string/numpy array/scipy.sparse
            Data source of _InnerDataset.
            When data type is string, it represents the path of txt file
        label : list or numpy 1-D array, optional
            Label of the data
        max_bin : int, required
            Max number of discrete bin for features
        reference : Other _InnerDataset, optional
            If this dataset validation, need to use training data as reference
        weight : list or numpy 1-D array , optional
            Weight for each instance.
        group : list or numpy 1-D array , optional
            Group/query size for dataset
        predictor : Predictor
            Used for continuned train
        silent : boolean, optional
            Whether print messages during construction
        feature_name : list of str
            Feature names
        categorical_feature : list of str or int
            Categorical features, type int represents index, \
            type str represents feature names (need to specify feature_name as well)
        params: dict, optional
            Other parameters
        """
        if data is None:
            self.handle = None
            return
        data = _data_from_pandas(data)
        label = _label_from_pandas(label)
        self.data_has_header = False
        """process for args"""
        params = {} if params is None else params
        self.max_bin = max_bin
        self.predictor = predictor
        params["max_bin"] = max_bin
        if silent:
            params["verbose"] = 0
        elif "verbose" not in params:
            params["verbose"] = 1
        """get categorical features"""
        if categorical_feature is not None:
            categorical_indices = set()
            feature_dict = {}
            if feature_name is not None:
                feature_dict = {name: i for i, name in enumerate(feature_name)}
            for name in categorical_feature:
                if is_str(name) and name in feature_dict:
                    categorical_indices.add(feature_dict[name])
                elif isinstance(name, int):
                    categorical_indices.add(name)
                else:
                    raise TypeError("unknown type({}) or unknown name({}) in categorical_feature" \
                        .format(type(name).__name__, name))

            params['categorical_column'] = categorical_indices

        params_str = param_dict_to_str(params)
        """process for reference dataset"""
        ref_dataset = None
        if isinstance(reference, _InnerDataset):
            ref_dataset = ctypes.byref(reference.handle)
        elif reference is not None:
            raise TypeError('Reference dataset should be None or dataset instance')
        """start construct data"""
        if is_str(data):
            """check data has header or not"""
            if params.get("has_header", "").lower() == "true" \
                or params.get("header", "").lower() == "true":
                self.data_has_header = True
            self.handle = ctypes.c_void_p()
            _safe_call(_LIB.LGBM_DatasetCreateFromFile(
                c_str(data),
                c_str(params_str),
                ref_dataset,
                ctypes.byref(self.handle)))
        elif isinstance(data, scipy.sparse.csr_matrix):
            self.__init_from_csr(data, params_str, ref_dataset)
        elif isinstance(data, np.ndarray):
            self.__init_from_np2d(data, params_str, ref_dataset)
        else:
            try:
                csr = scipy.sparse.csr_matrix(data)
                self.__init_from_csr(csr, params_str, ref_dataset)
            except:
                raise TypeError('can not initialize _InnerDataset from {}'.format(type(data).__name__))
        if label is not None:
            self.set_label(label)
        if self.get_label() is None:
            raise ValueError("label should not be None")
        if weight is not None:
            self.set_weight(weight)
        if group is not None:
            self.set_group(group)
        # load init score
        if self.predictor is not None and isinstance(self.predictor, Predictor):
            init_score = self.predictor.predict(data,
                                                raw_score=True,
                                                data_has_header=self.data_has_header,
                                                is_reshape=False)
            if self.predictor.num_class > 1:
                # need re group init score
                new_init_score = np.zeros(init_score.size(), dtype=np.float32)
                num_data = self.num_data()
                for i in range(num_data):
                    for j in range(self.predictor.num_class):
                        new_init_score[j * num_data + i] = init_score[i * self.predictor.num_class + j]
                init_score = new_init_score
            self.set_init_score(init_score)
        # set feature names
        self.set_feature_name(feature_name)

    def create_valid(self, data, label=None, weight=None, group=None,
                     silent=False, params=None):
        """
        Create validation data align with current dataset

        Parameters
        ----------
        data : string/numpy array/scipy.sparse
            Data source of _InnerDataset.
            When data type is string, it represents the path of txt file
        label : list or numpy 1-D array, optional
            Label of the training data.
        weight : list or numpy 1-D array , optional
            Weight for each instance.
        group : list or numpy 1-D array , optional
            Group/query size for dataset
        silent : boolean, optional
            Whether print messages during construction
        params: dict, optional
            Other parameters
        """
        return _InnerDataset(data, label=label, max_bin=self.max_bin, reference=self,
                       weight=weight, group=group, predictor=self.predictor,
                       silent=silent, params=params)

    def subset(self, used_indices, params=None):
        """
        Get subset of current dataset
        """
        used_indices = list_to_1d_numpy(used_indices, np.int32)
        ret = _InnerDataset(None)
        ret.handle = ctypes.c_void_p()
        params_str = param_dict_to_str(params)
        _safe_call(_LIB.LGBM_DatasetGetSubset(
            ctypes.byref(self.handle),
            used_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            used_indices.shape[0],
            c_str(params_str),
            ctypes.byref(ret.handle)))
        ret.max_bin = self.max_bin
        ret.predictor = self.predictor
        if ret.get_label() is None:
            raise ValueError("label should not be None")
        return ret

    def set_feature_name(self, feature_name):
        if feature_name is None:
            return
        if len(feature_name) != self.num_feature():
            raise ValueError("size of feature_name error")
        c_feature_name = [c_str(name) for name in feature_name]
        _safe_call(_LIB.LGBM_DatasetSetFeatureNames(
            self.handle,
            c_array(ctypes.c_char_p, c_feature_name),
            len(feature_name)))

    def __init_from_np2d(self, mat, params_str, ref_dataset):
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
        _safe_call(_LIB.LGBM_DatasetCreateFromMat(
            ptr_data,
            type_ptr_data,
            mat.shape[0],
            mat.shape[1],
            C_API_IS_ROW_MAJOR,
            c_str(params_str),
            ref_dataset,
            ctypes.byref(self.handle)))

    def __init_from_csr(self, csr, params_str, ref_dataset):
        """
        Initialize data from a CSR matrix.
        """
        if len(csr.indices) != len(csr.data):
            raise ValueError('length mismatch: {} vs {}'.format(len(csr.indices), len(csr.data)))
        self.handle = ctypes.c_void_p()

        ptr_indptr, type_ptr_indptr = c_int_array(csr.indptr)
        ptr_data, type_ptr_data = c_float_array(csr.data)

        _safe_call(_LIB.LGBM_DatasetCreateFromCSR(
            ptr_indptr,
            type_ptr_indptr,
            csr.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ptr_data,
            type_ptr_data,
            len(csr.indptr),
            len(csr.data),
            csr.shape[1],
            c_str(params_str),
            ref_dataset,
            ctypes.byref(self.handle)))

    def __del__(self):
        _safe_call(_LIB.LGBM_DatasetFree(self.handle))

    def get_field(self, field_name):
        """Get property from the _InnerDataset.

        Parameters
        ----------
        field_name: str
            The field name of the information

        Returns
        -------
        info : array
            A numpy array of information of the data
        """
        tmp_out_len = ctypes.c_int64()
        out_type = ctypes.c_int32()
        ret = ctypes.POINTER(ctypes.c_void_p)()
        _safe_call(_LIB.LGBM_DatasetGetField(
            self.handle,
            c_str(field_name),
            ctypes.byref(tmp_out_len),
            ctypes.byref(ret),
            ctypes.byref(out_type)))
        if out_type.value != FIELD_TYPE_MAPPER[field_name]:
            raise TypeError("Return type error for get_field")
        if tmp_out_len.value == 0:
            return None
        if out_type.value == C_API_DTYPE_INT32:
            return cint32_array_to_numpy(ctypes.cast(ret, ctypes.POINTER(ctypes.c_int32)), tmp_out_len.value)
        elif out_type.value == C_API_DTYPE_FLOAT32:
            return cfloat32_array_to_numpy(ctypes.cast(ret, ctypes.POINTER(ctypes.c_float)), tmp_out_len.value)
        else:
            raise TypeError("unknow type")

    def set_field(self, field_name, data):
        """Set property into the _InnerDataset.

        Parameters
        ----------
        field_name: str
            The field name of the information

        data: numpy array or list or None
            The array ofdata to be set
        """
        if data is None:
            """set to None"""
            _safe_call(_LIB.LGBM_DatasetSetField(
                self.handle,
                c_str(field_name),
                None,
                0,
                FIELD_TYPE_MAPPER[field_name]))
            return
        if IS_PANDAS_INSTALLED and isinstance(data, Series):
            dtype = np.int32 if field_name == 'group' else np.float32
            data = data.astype(dtype).values
        if not is_numpy_1d_array(data):
            raise TypeError("Unknow type({})".format(type(data).__name__))
        if data.dtype == np.float32:
            ptr_data = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            type_data = C_API_DTYPE_FLOAT32
        elif data.dtype == np.int32:
            ptr_data = data.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            type_data = C_API_DTYPE_INT32
        else:
            raise TypeError("excepted np.float32 or np.int32, met type({})".format(data.dtype))
        if type_data != FIELD_TYPE_MAPPER[field_name]:
            raise TypeError("type error for set_field")
        _safe_call(_LIB.LGBM_DatasetSetField(
            self.handle,
            c_str(field_name),
            ptr_data,
            len(data),
            type_data))

    def save_binary(self, filename):
        """Save _InnerDataset to binary file

        Parameters
        ----------
        filename : string
            Name of the output file.
        """
        _safe_call(_LIB.LGBM_DatasetSaveBinary(
            self.handle,
            c_str(filename)))

    def set_label(self, label):
        """Set label of _InnerDataset

        Parameters
        ----------
        label: numpy array or list or None
            The label information to be set into _InnerDataset
        """
        label = list_to_1d_numpy(label, np.float32)
        self.set_field('label', label)

    def set_weight(self, weight):
        """ Set weight of each instance.

        Parameters
        ----------
        weight : numpy array or list or None
            Weight for each data point
        """
        if weight is not None:
            weight = list_to_1d_numpy(weight, np.float32)
        self.set_field('weight', weight)

    def set_init_score(self, score):
        """ Set init score of booster to start from.

        Parameters
        ----------
        score: numpy array or list or None
            Init score for booster
        """
        if score is not None:
            score = list_to_1d_numpy(score, np.float32)
        self.set_field('init_score', score)

    def set_group(self, group):
        """Set group size of _InnerDataset (used for ranking).

        Parameters
        ----------
        group : numpy array or list or None
            Group size of each group
        """
        if group is not None:
            group = list_to_1d_numpy(group, np.int32)
        self.set_field('group', group)

    def get_label(self):
        """Get the label of the _InnerDataset.

        Returns
        -------
        label : array
        """
        return self.get_field('label')

    def get_weight(self):
        """Get the weight of the _InnerDataset.

        Returns
        -------
        weight : array
        """
        return self.get_field('weight')

    def get_init_score(self):
        """Get the initial score of the _InnerDataset.

        Returns
        -------
        init_score : array
        """
        return self.get_field('init_score')

    def get_group(self):
        """Get the initial score of the _InnerDataset.

        Returns
        -------
        init_score : array
        """
        return self.get_field('group')

    def num_data(self):
        """Get the number of rows in the _InnerDataset.

        Returns
        -------
        number of rows : int
        """
        ret = ctypes.c_int64()
        _safe_call(_LIB.LGBM_DatasetGetNumData(self.handle,
                                               ctypes.byref(ret)))
        return ret.value

    def num_feature(self):
        """Get the number of columns (features) in the _InnerDataset.

        Returns
        -------
        number of columns : int
        """
        ret = ctypes.c_int64()
        _safe_call(_LIB.LGBM_DatasetGetNumFeature(self.handle,
                                                  ctypes.byref(ret)))
        return ret.value

class Dataset(object):
    """High level Dataset used in LightGBM.
    """

    def __init__(self, data, label=None, max_bin=255, reference=None,
                 weight=None, group=None, predictor=None, silent=False,
                 feature_name=None, categorical_feature=None, params=None,
                 free_raw_data=True, lazy_init=True):
        """
        Dataset used in LightGBM.

        Parameters
        ----------
        data : string/numpy array/scipy.sparse
            Data source of Dataset.
            When data type is string, it represents the path of txt file
        label : list or numpy 1-D array, optional
            Label of the data
        max_bin : int, required
            Max number of discrete bin for features
        reference : Other Dataset, optional
            If this dataset validation, need to use training data as reference
        weight : list or numpy 1-D array , optional
            Weight for each instance.
        group : list or numpy 1-D array , optional
            Group/query size for dataset
        predictor : Predictor
            Used for continuned train
        silent : boolean, optional
            Whether print messages during construction
        feature_name : list of str
            Feature names
        categorical_feature : list of str or int
            Categorical features, type int represents index, \
            type str represents feature names (need to specify feature_name as well)
        params: dict, optional
            Other parameters
        free_raw_data: Bool
            True if need to free raw data after construct inner dataset
        lazy_init: Bool
            False if need to construct immediately
        """
        self.data = data
        self.label = label
        self.max_bin = max_bin
        self.reference = reference
        self.weight = weight
        self.group = group
        self.predictor = predictor
        self.silent = silent
        self.feature_name = feature_name
        self.categorical_feature = categorical_feature
        self.params = params
        self.free_raw_data = free_raw_data
        self.inner_dataset = None
        self.used_indices = None
        self.lazy_init = lazy_init
        if not self.lazy_init:
            self.construct()

    def create_valid(self, data, label=None, weight=None, group=None,
                     silent=False, params=None):
        """
        Create validation data align with current dataset

        Parameters
        ----------
        data : string/numpy array/scipy.sparse
            Data source of _InnerDataset.
            When data type is string, it represents the path of txt file
        label : list or numpy 1-D array, optional
            Label of the training data.
        weight : list or numpy 1-D array , optional
            Weight for each instance.
        group : list or numpy 1-D array , optional
            Group/query size for dataset
        silent : boolean, optional
            Whether print messages during construction
        params: dict, optional
            Other parameters
        """
        return Dataset(data, label=label, max_bin=self.max_bin, reference=self,
                       weight=weight, group=group, predictor=self.predictor,
                       silent=silent, params=params, free_raw_data=self.free_raw_data,
                       lazy_init=self.lazy_init)

    def construct(self):
        """Lazy init"""
        if self.inner_dataset is None:
            if self.reference is not None:
                if self.used_indices is None:
                    self.inner_dataset = self.reference._get_inner_dataset().create_valid(
                        self.data, self.label,
                        self.weight, self.group,
                        self.silent, self.params)
                else:
                    """construct subset"""
                    self.inner_dataset = self.reference._get_inner_dataset().subset(
                        self.used_indices, self.params)
            else:
                self.inner_dataset = _InnerDataset(self.data, self.label, self.max_bin,
                    None, self.weight, self.group, self.predictor,
                    self.silent, self.feature_name, self.categorical_feature, self.params)
            if self.free_raw_data:
                self.data = None

    def _get_inner_dataset(self):
        self.construct()
        return self.inner_dataset

    def __is_constructed(self):
        return self.inner_dataset is not None

    def set_categorical_feature(self, categorical_feature):
        if self.categorical_feature == categorical_feature:
            return
        if self.data is not None:
            self.categorical_feature = categorical_feature
            self.inner_dataset = None
        else:
            raise Exception("Cannot set categorical feature after freed raw data")

    def set_predictor(self, predictor):
        if predictor is self.predictor:
            return
        if self.data is not None:
            self.predictor = predictor
            self.inner_dataset = None
        else:
            raise Exception("Cannot set predictor after freed raw data")

    def set_reference(self, reference):
        if self.reference is reference:
            return
        if self.data is not None:
            self.reference = reference
            self.inner_dataset = None
        else:
            raise Exception("Cannot set reference after freed raw data")

    def set_feature_name(self, feature_name):
        self.feature_name = feature_name
        if self.__is_constructed():
            self.inner_dataset.set_feature_name(self.feature_name)

    def subset(self, used_indices, params=None, lazy_init=True):
        """
        Get subset of current dataset
        """
        ret = Dataset(None)
        ret.reference = self
        ret.used_indices = used_indices
        ret.params = param
        if not lazy_init:
            ret.construct()
        return ret

    def save_binary(self, filename):
        """Save Dataset to binary file

        Parameters
        ----------
        filename : string
            Name of the output file.
        """
        self._get_inner_dataset().save_binary(filename)


    def set_label(self, label):
        """Set label of Dataset

        Parameters
        ----------
        label: numpy array or list or None
            The label information to be set into Dataset
        """
        self.label = label
        if self.__is_constructed():
            self.inner_dataset.set_label(self.label)

    def set_weight(self, weight):
        """ Set weight of each instance.

        Parameters
        ----------
        weight : numpy array or list or None
            Weight for each data point
        """
        self.weight = weight
        if self.__is_constructed():
            self.inner_dataset.set_weight(self.weight)

    def set_init_score(self, init_score):
        """ Set init score of booster to start from.

        Parameters
        ----------
        init_score: numpy array or list or None
            Init score for booster
        """
        self.init_score = init_score
        if self.__is_constructed():
            self.inner_dataset.set_init_score(self.init_score)

    def set_group(self, group):
        """Set group size of Dataset (used for ranking).

        Parameters
        ----------
        group : numpy array or list or None
            Group size of each group
        """
        self.group = group
        if self.__is_constructed():
            self.inner_dataset.set_group(self.group)

    def get_label(self):
        """Get the label of the Dataset.

        Returns
        -------
        label : array
        """
        if self.label is None and self.__is_constructed():
            self.label = self.inner_dataset.get_label()
        return self.label

    def get_weight(self):
        """Get the weight of the Dataset.

        Returns
        -------
        weight : array
        """
        if self.weight is None and self.__is_constructed():
            self.weight = self.inner_dataset.get_weight()
        return self.weight

    def get_init_score(self):
        """Get the initial score of the Dataset.

        Returns
        -------
        init_score : array
        """
        if self.init_score is None and self.__is_constructed():
            self.init_score = self.inner_dataset.get_init_score()
        return self.init_score

    def get_group(self):
        """Get the initial score of the Dataset.

        Returns
        -------
        init_score : array
        """
        if self.group is None and self.__is_constructed():
            self.group = self.inner_dataset.get_group()
        return self.group

    def num_data(self):
        """Get the number of rows in the Dataset.

        Returns
        -------
        number of rows : int
        """
        if self.__is_constructed():
            return self.inner_dataset.num_data()
        else:
            raise Exception("Cannot call num_data before construct, please call it explicitly")

    def num_feature(self):
        """Get the number of columns (features) in the Dataset.

        Returns
        -------
        number of columns : int
        """
        if self.__is_constructed():
            return self.inner_dataset.num_feature()
        else:
            raise Exception("Cannot call num_feature before construct, please call it explicitly")

class Booster(object):
    """"A Booster of LightGBM.
    """
    def __init__(self, params=None, train_set=None, model_file=None, silent=False):
        """Initialize the Booster.

        Parameters
        ----------
        params : dict
            Parameters for boosters.
        train_set : Dataset
            Training dataset
        model_file : string
            Path to the model file.
        silent : boolean, optional
            Whether print messages during construction
        """
        self.handle = ctypes.c_void_p()
        self.__need_reload_eval_info = True
        self.__is_manage_handle = True
        self.__train_data_name = "training"
        self.__attr = {}
        self.best_iteration = -1
        params = {} if params is None else params
        if silent:
            params["verbose"] = 0
        elif "verbose" not in params:
            params["verbose"] = 1
        if train_set is not None:
            """Training task"""
            if not isinstance(train_set, Dataset):
                raise TypeError('training data should be Dataset instance, met {}'.format(type(train_set).__name__))
            params_str = param_dict_to_str(params)
            """construct booster object"""
            _safe_call(_LIB.LGBM_BoosterCreate(
                train_set._get_inner_dataset().handle,
                c_str(params_str),
                ctypes.byref(self.handle)))
            """save reference to data"""
            self.train_set = train_set
            self.valid_sets = []
            self.name_valid_sets = []
            self.__num_dataset = 1
            self.init_predictor = train_set.predictor
            if self.init_predictor is not None:
                _safe_call(_LIB.LGBM_BoosterMerge(
                    self.handle,
                    self.init_predictor.handle))
            out_num_class = ctypes.c_int64(0)
            _safe_call(_LIB.LGBM_BoosterGetNumClasses(
                self.handle,
                ctypes.byref(out_num_class)))
            self.__num_class = out_num_class.value
            """buffer for inner predict"""
            self.__inner_predict_buffer = [None]
            self.__is_predicted_cur_iter = [False]
            self.__get_eval_info()
        elif model_file is not None:
            """Prediction task"""
            out_num_iterations = ctypes.c_int64(0)
            _safe_call(_LIB.LGBM_BoosterCreateFromModelfile(
                c_str(model_file),
                ctypes.byref(out_num_iterations),
                ctypes.byref(self.handle)))
            out_num_class = ctypes.c_int64(0)
            _safe_call(_LIB.LGBM_BoosterGetNumClasses(
                self.handle,
                ctypes.byref(out_num_class)))
            self.__num_class = out_num_class.value
        else:
            raise TypeError('At least need training dataset or model file to create booster instance')

    def __del__(self):
        if self.handle is not None and self.__is_manage_handle:
            _safe_call(_LIB.LGBM_BoosterFree(self.handle))

    def set_train_data_name(self, name):
        self.__train_data_name = name

    def add_valid(self, data, name):
        """Add an validation data

        Parameters
        ----------
        data : Dataset
            Validation data
        name : String
            Name of validation data
        """
        if data.predictor is not self.init_predictor:
            raise Exception("Add validation data failed, you should use same predictor for these data")
        _safe_call(_LIB.LGBM_BoosterAddValidData(
            self.handle,
            data._get_inner_dataset().handle))
        self.valid_sets.append(data)
        self.name_valid_sets.append(name)
        self.__num_dataset += 1
        self.__inner_predict_buffer.append(None)
        self.__is_predicted_cur_iter.append(False)

    def reset_parameter(self, params):
        """Reset parameters for booster

        Parameters
        ----------
        params : dict
            New parameters for boosters
        silent : boolean, optional
            Whether print messages during construction
        """
        if 'metric' in params:
            self.__need_reload_eval_info = True
        params_str = param_dict_to_str(params)
        if params_str:
            _safe_call(_LIB.LGBM_BoosterResetParameter(
                self.handle,
                c_str(params_str)))

    def update(self, train_set=None, fobj=None):
        """
        Update for one iteration
        Note: for multi-class task, the score is group by class_id first, then group by row_id
              if you want to get i-th row score in j-th class, the access way is score[j*num_data+i]
              and you should group grad and hess in this way as well

        Parameters
        ----------
        train_set :
            Training data, None means use last training data
        fobj : function
            Customized objective function.

        Returns
        -------
        is_finished, bool
        """

        """need reset training data"""
        if train_set is not None and train_set is not self.train_set:
            if train_set.predictor is not self.init_predictor:
                raise Exception("Replace training data failed, you should use same predictor for these data")
            self.train_set = train_set
            _safe_call(_LIB.LGBM_BoosterResetTrainingData(
                self.handle,
                self.train_set._get_inner_dataset().handle))
            self.__inner_predict_buffer[0] = None
        is_finished = ctypes.c_int(0)
        if fobj is None:
            _safe_call(_LIB.LGBM_BoosterUpdateOneIter(
                self.handle,
                ctypes.byref(is_finished)))
            self.__is_predicted_cur_iter = [False for _ in range(self.__num_dataset)]
            return is_finished.value == 1
        else:
            grad, hess = fobj(self.__inner_predict(0), self.train_set)
            return self.__boost(grad, hess)

    def __boost(self, grad, hess):
        """
        Boost the booster for one iteration, with customized gradient statistics.
        Note: for multi-class task, the score is group by class_id first, then group by row_id
              if you want to get i-th row score in j-th class, the access way is score[j*num_data+i]
              and you should group grad and hess in this way as well

        Parameters
        ----------
        grad : 1d numpy or 1d list
            The first order of gradient.
        hess : 1d numpy or 1d list
            The second order of gradient.

        Returns
        -------
        is_finished, bool
        """
        if not is_numpy_1d_array(grad):
            if is_1d_list(grad):
                grad = np.array(grad, dtype=np.float32, copy=False)
            else:
                raise TypeError("grad should be numpy 1d array or 1d list")
        if not is_numpy_1d_array(hess):
            if is_1d_list(hess):
                hess = np.array(hess, dtype=np.float32, copy=False)
            else:
                raise TypeError("hess should be numpy 1d array or 1d list")
        if len(grad) != len(hess):
            raise ValueError('grad / hess lengths mismatch: {} / {}'.format(len(grad), len(hess)))
        if grad.dtype != np.float32:
            grad = grad.astype(np.float32, copy=False)
        if hess.dtype != np.float32:
            hess = hess.astype(np.float32, copy=False)
        is_finished = ctypes.c_int(0)
        _safe_call(_LIB.LGBM_BoosterUpdateOneIterCustom(
            self.handle,
            grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            hess.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(is_finished)))
        self.__is_predicted_cur_iter = [False for _ in range(self.__num_dataset)]
        return is_finished.value == 1

    def rollback_one_iter(self):
        """
        Rollback one iteration
        """
        _safe_call(_LIB.LGBM_BoosterRollbackOneIter(
            self.handle))
        self.__is_predicted_cur_iter = [False for _ in range(self.__num_dataset)]

    def current_iteration(self):
        out_cur_iter = ctypes.c_int64(0)
        _safe_call(_LIB.LGBM_BoosterGetCurrentIteration(
            self.handle,
            ctypes.byref(out_cur_iter)))
        return out_cur_iter.value

    def eval(self, data, name, feval=None):
        """Evaluate for data

        Parameters
        ----------
        data : _InnerDataset object
        name :
            Name of data
        feval : function
            Custom evaluation function.
        Returns
        -------
        result: list
            Evaluation result list.
        """
        if not isinstance(data, _InnerDataset):
            raise TypeError("Can only eval for _InnerDataset instance")
        data_idx = -1
        if data is self.train_set:
            data_idx = 0
        else:
            for i in range(len(self.valid_sets)):
                if data is self.valid_sets[i]:
                    data_idx = i + 1
                    break
        """need to push new valid data"""
        if data_idx == -1:
            self.add_valid(data, name)
            data_idx = self.__num_dataset - 1

        return self.__inner_eval(name, data_idx, feval)

    def eval_train(self, feval=None):
        """Evaluate for training data

        Parameters
        ----------
        feval : function
            Custom evaluation function.

        Returns
        -------
        result: str
            Evaluation result list.
        """
        return self.__inner_eval(self.__train_data_name, 0, feval)

    def eval_valid(self, feval=None):
        """Evaluate for validation data

        Parameters
        ----------
        feval : function
            Custom evaluation function.

        Returns
        -------
        result: str
            Evaluation result list.
        """
        return [item for i in range(1, self.__num_dataset) \
            for item in self.__inner_eval(self.name_valid_sets[i-1], i, feval)]

    def save_model(self, filename, num_iteration=-1):
        """Save model of booster to file

        Parameters
        ----------
        filename : str
            Filename to save
        num_iteration: int
            Number of iteration that want to save. < 0 means save all
        """
        _safe_call(_LIB.LGBM_BoosterSaveModel(
            self.handle,
            num_iteration,
            c_str(filename)))

    def dump_model(self):
        """Dump model to json format

        Returns
        -------
        Json format of model
        """
        buffer_len = 1 << 20
        tmp_out_len = ctypes.c_int64(0)
        string_buffer = ctypes.create_string_buffer(buffer_len)
        ptr_string_buffer = ctypes.c_char_p(*[ctypes.addressof(string_buffer)])
        _safe_call(_LIB.LGBM_BoosterDumpModel(
            self.handle,
            buffer_len,
            ctypes.byref(tmp_out_len),
            ctypes.byref(ptr_string_buffer)))
        actual_len = tmp_out_len.value
        '''if buffer length is not long enough, reallocate a buffer'''
        if actual_len > buffer_len:
            string_buffer = ctypes.create_string_buffer(actual_len)
            ptr_string_buffer = ctypes.c_char_p(*[ctypes.addressof(string_buffer)])
            _safe_call(_LIB.LGBM_BoosterDumpModel(
                self.handle,
                actual_len,
                ctypes.byref(tmp_out_len),
                ctypes.byref(ptr_string_buffer)))
        return json.loads(string_buffer.value.decode())

    def predict(self, data, num_iteration=-1, raw_score=False, pred_leaf=False, data_has_header=False, is_reshape=True):
        """Predict logic

        Parameters
        ----------
        data : string/numpy array/scipy.sparse
            Data source for prediction
            When data type is string, it represents the path of txt file
        num_iteration : int
            Used iteration for prediction
        raw_score : bool
            True for predict raw score
        pred_leaf : bool
            True for predict leaf index
        data_has_header : bool
            Used for txt data
        is_reshape : bool
            Reshape to (nrow, ncol) if true

        Returns
        -------
        Prediction result
        """
        predictor = Predictor(booster_handle=self.handle, is_manage_handle=False)
        return predictor.predict(data, num_iteration, raw_score, pred_leaf, data_has_header, is_reshape)

    def to_predictor(self):
        """Convert to predictor
        Note: Predictor will manage the handle after doing this
        """
        predictor = Predictor(booster_handle=self.handle, is_manage_handle=True)
        self.__is_manage_handle = False
        return predictor

    def feature_importance(self, importance_type='split'):
        """Feature importances

        Returns
        -------
        Array of feature importances
        """
        if importance_type not in ["split", "gain"]:
            raise KeyError("importance_type must be split or gain")
        dump_model = self.dump_model()
        ret = [0] * (dump_model["max_feature_idx"] + 1)
        def dfs(root):
            if "split_feature" in root:
                if importance_type == 'split':
                    ret[root["split_feature"]] += 1
                elif importance_type == 'gain':
                    ret[root["split_feature"]] += root["split_gain"]
                dfs(root["left_child"])
                dfs(root["right_child"])
        for tree in dump_model["tree_info"]:
            dfs(tree["tree_structure"])
        return np.array(ret)

    def __inner_eval(self, data_name, data_idx, feval=None):
        """
        Evaulate training or validation data
        """
        if data_idx >= self.__num_dataset:
            raise ValueError("data_idx should be smaller than number of dataset")
        self.__get_eval_info()
        ret = []
        if self.__num_inner_eval > 0:
            result = np.array([0.0 for _ in range(self.__num_inner_eval)], dtype=np.float32)
            tmp_out_len = ctypes.c_int64(0)
            _safe_call(_LIB.LGBM_BoosterGetEval(
                self.handle,
                data_idx,
                ctypes.byref(tmp_out_len),
                result.ctypes.data_as(ctypes.POINTER(ctypes.c_float))))
            if tmp_out_len.value != self.__num_inner_eval:
                raise ValueError("incorrect number of eval results")
            for i in range(self.__num_inner_eval):
                ret.append((data_name, self.__name_inner_eval[i], result[i], self.__higher_better_inner_eval[i]))
        if feval is not None:
            if data_idx == 0:
                cur_data = self.train_set
            else:
                cur_data = self.valid_sets[data_idx - 1]
            feval_ret = feval(self.__inner_predict(data_idx), cur_data)
            if isinstance(feval_ret, list):
                for eval_name, val, is_higher_better in feval_ret:
                    ret.append((data_name, eval_name, val, is_higher_better))
            else:
                eval_name, val, is_higher_better = feval_ret
                ret.append((data_name, eval_name, val, is_higher_better))
        return ret

    def __inner_predict(self, data_idx):
        """
        Predict for training and validation dataset
        """
        if data_idx >= self.__num_dataset:
            raise ValueError("data_idx should be smaller than number of dataset")
        if self.__inner_predict_buffer[data_idx] is None:
            if data_idx == 0:
                n_preds = self.train_set.num_data() * self.__num_class
            else:
                n_preds = self.valid_sets[data_idx - 1].num_data() * self.__num_class
            self.__inner_predict_buffer[data_idx] = \
                np.array([0.0 for _ in range(n_preds)], dtype=np.float32, copy=False)
        """avoid to predict many time in one iteration"""
        if not self.__is_predicted_cur_iter[data_idx]:
            tmp_out_len = ctypes.c_int64(0)
            data_ptr = self.__inner_predict_buffer[data_idx].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            _safe_call(_LIB.LGBM_BoosterGetPredict(
                self.handle,
                data_idx,
                ctypes.byref(tmp_out_len),
                data_ptr))
            if tmp_out_len.value != len(self.__inner_predict_buffer[data_idx]):
                raise ValueError("incorrect number of predict results for data %d" % (data_idx))
            self.__is_predicted_cur_iter[data_idx] = True
        return self.__inner_predict_buffer[data_idx]

    def __get_eval_info(self):
        """
        Get inner evaluation count and names
        """
        if self.__need_reload_eval_info:
            self.__need_reload_eval_info = False
            out_num_eval = ctypes.c_int64(0)
            """Get num of inner evals"""
            _safe_call(_LIB.LGBM_BoosterGetEvalCounts(
                self.handle,
                ctypes.byref(out_num_eval)))
            self.__num_inner_eval = out_num_eval.value
            if self.__num_inner_eval > 0:
                """Get name of evals"""
                tmp_out_len = ctypes.c_int64(0)
                string_buffers = [ctypes.create_string_buffer(255) for i in range(self.__num_inner_eval)]
                ptr_string_buffers = (ctypes.c_char_p*self.__num_inner_eval)(*map(ctypes.addressof, string_buffers))
                _safe_call(_LIB.LGBM_BoosterGetEvalNames(
                    self.handle,
                    ctypes.byref(tmp_out_len),
                    ptr_string_buffers))
                if self.__num_inner_eval != tmp_out_len.value:
                    raise ValueError("size of eval names doesn't equal with num_evals")
                self.__name_inner_eval = \
                    [string_buffers[i].value.decode() for i in range(self.__num_inner_eval)]
                self.__higher_better_inner_eval = \
                    [name.startswith(('auc', 'ndcg')) for name in self.__name_inner_eval]

    def attr(self, key):
        """Get attribute string from the Booster.

        Parameters
        ----------
        key : str
            The key to get attribute from.

        Returns
        -------
        value : str
            The attribute value of the key, returns None if attribute do not exist.
        """
        return self.__attr.get(key, None)

    def set_attr(self, **kwargs):
        """Set the attribute of the Booster.

        Parameters
        ----------
        **kwargs
            The attributes to set. Setting a value to None deletes an attribute.
        """
        for key, value in kwargs.items():
            if value is not None:
                if not is_str(value):
                    raise ValueError("set_attr only accepts string values")
                self.__attr[key] = value
            else:
                self.__attr.pop(key, None)
