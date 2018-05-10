# coding: utf-8
# pylint: disable = invalid-name, C0111, C0301
# pylint: disable = R0912, R0913, R0914, W0105, W0201, W0212
"""Wrapper c_api of LightGBM"""
from __future__ import absolute_import

import copy
import ctypes
import os
import warnings
from tempfile import NamedTemporaryFile

import numpy as np
import scipy.sparse

from .compat import (DataFrame, LGBMDeprecationWarning, Series,
                     decode_string, integer_types,
                     json, json_default_with_numpy,
                     numeric_types, range_, string_type)
from .libpath import find_lib_path


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
        raise LightGBMError(decode_string(_LIB.LGBM_GetLastError()))


def is_numeric(obj):
    """Check is a number or not, include numpy number etc."""
    try:
        float(obj)
        return True
    except (TypeError, ValueError):
        # TypeError: obj is not a string or a number
        # ValueError: invalid literal
        return False


def is_numpy_1d_array(data):
    """Check is 1d numpy array"""
    return isinstance(data, np.ndarray) and len(data.shape) == 1


def is_1d_list(data):
    """Check is 1d list"""
    return isinstance(data, list) and \
        (not data or is_numeric(data[0]))


def list_to_1d_numpy(data, dtype=np.float32, name='list'):
    """convert to 1d numpy array"""
    if is_numpy_1d_array(data):
        if data.dtype == dtype:
            return data
        else:
            return data.astype(dtype=dtype, copy=False)
    elif is_1d_list(data):
        return np.array(data, dtype=dtype, copy=False)
    elif isinstance(data, Series):
        return data.values.astype(dtype)
    else:
        raise TypeError("Wrong type({}) for {}, should be list or numpy array".format(type(data).__name__, name))


def cfloat32_array_to_numpy(cptr, length):
    """Convert a ctypes float pointer array to a numpy array.
    """
    if isinstance(cptr, ctypes.POINTER(ctypes.c_float)):
        return np.fromiter(cptr, dtype=np.float32, count=length)
    else:
        raise RuntimeError('Expected float pointer')


def cfloat64_array_to_numpy(cptr, length):
    """Convert a ctypes double pointer array to a numpy array.
    """
    if isinstance(cptr, ctypes.POINTER(ctypes.c_double)):
        return np.fromiter(cptr, dtype=np.float64, count=length)
    else:
        raise RuntimeError('Expected double pointer')


def cint32_array_to_numpy(cptr, length):
    """Convert a ctypes float pointer array to a numpy array.
    """
    if isinstance(cptr, ctypes.POINTER(ctypes.c_int32)):
        return np.fromiter(cptr, dtype=np.int32, count=length)
    else:
        raise RuntimeError('Expected int pointer')


def c_str(string):
    """Convert a python string to cstring."""
    return ctypes.c_char_p(string.encode('utf-8'))


def c_array(ctype, values):
    """Convert a python array to c array."""
    return (ctype * len(values))(*values)


def param_dict_to_str(data):
    if data is None or not data:
        return ""
    pairs = []
    for key, val in data.items():
        if isinstance(val, (list, tuple, set)) or is_numpy_1d_array(val):
            pairs.append(str(key) + '=' + ','.join(map(str, val)))
        elif isinstance(val, string_type) or isinstance(val, numeric_types) or is_numeric(val):
            pairs.append(str(key) + '=' + str(val))
        elif val is not None:
            raise TypeError('Unknown type of parameter:%s, got:%s'
                            % (key, type(val).__name__))
    return ' '.join(pairs)


class _temp_file(object):
    def __enter__(self):
        with NamedTemporaryFile(prefix="lightgbm_tmp_", delete=True) as f:
            self.name = f.name
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if os.path.isfile(self.name):
            os.remove(self.name)

    def readlines(self):
        with open(self.name, "r+") as f:
            ret = f.readlines()
        return ret

    def writelines(self, lines):
        with open(self.name, "w+") as f:
            f.writelines(lines)


"""marco definition of data type in c_api of LightGBM"""
C_API_DTYPE_FLOAT32 = 0
C_API_DTYPE_FLOAT64 = 1
C_API_DTYPE_INT32 = 2
C_API_DTYPE_INT64 = 3

"""Matric is row major in python"""
C_API_IS_ROW_MAJOR = 1

"""marco definition of prediction type in c_api of LightGBM"""
C_API_PREDICT_NORMAL = 0
C_API_PREDICT_RAW_SCORE = 1
C_API_PREDICT_LEAF_INDEX = 2
C_API_PREDICT_CONTRIB = 3

"""data type of data field"""
FIELD_TYPE_MAPPER = {"label": C_API_DTYPE_FLOAT32,
                     "weight": C_API_DTYPE_FLOAT32,
                     "init_score": C_API_DTYPE_FLOAT64,
                     "group": C_API_DTYPE_INT32}


def convert_from_sliced_object(data):
    """fix the memory of multi-dimensional sliced object"""
    if data.base is not None and isinstance(data, np.ndarray) and isinstance(data.base, np.ndarray):
        if not data.flags.c_contiguous:
            warnings.warn("Usage subset(sliced data) of np.ndarray is not recommended due to it will double the peak memory cost in LightGBM.")
            return np.copy(data)
    return data


def c_float_array(data):
    """get pointer of float numpy array / list"""
    if is_1d_list(data):
        data = np.array(data, copy=False)
    if is_numpy_1d_array(data):
        data = convert_from_sliced_object(data)
        assert data.flags.c_contiguous
        if data.dtype == np.float32:
            ptr_data = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            type_data = C_API_DTYPE_FLOAT32
        elif data.dtype == np.float64:
            ptr_data = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            type_data = C_API_DTYPE_FLOAT64
        else:
            raise TypeError("Expected np.float32 or np.float64, met type({})"
                            .format(data.dtype))
    else:
        raise TypeError("Unknown type({})".format(type(data).__name__))
    return (ptr_data, type_data, data)  # return `data` to avoid the temporary copy is freed


def c_int_array(data):
    """get pointer of int numpy array / list"""
    if is_1d_list(data):
        data = np.array(data, copy=False)
    if is_numpy_1d_array(data):
        data = convert_from_sliced_object(data)
        assert data.flags.c_contiguous
        if data.dtype == np.int32:
            ptr_data = data.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            type_data = C_API_DTYPE_INT32
        elif data.dtype == np.int64:
            ptr_data = data.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
            type_data = C_API_DTYPE_INT64
        else:
            raise TypeError("Expected np.int32 or np.int64, met type({})"
                            .format(data.dtype))
    else:
        raise TypeError("Unknown type({})".format(type(data).__name__))
    return (ptr_data, type_data, data)  # return `data` to avoid the temporary copy is freed


PANDAS_DTYPE_MAPPER = {'int8': 'int', 'int16': 'int', 'int32': 'int',
                       'int64': 'int', 'uint8': 'int', 'uint16': 'int',
                       'uint32': 'int', 'uint64': 'int', 'float16': 'float',
                       'float32': 'float', 'float64': 'float', 'bool': 'int'}


def _data_from_pandas(data, feature_name, categorical_feature, pandas_categorical):
    if isinstance(data, DataFrame):
        if feature_name == 'auto' or feature_name is None:
            data = data.rename(columns=str)
        cat_cols = data.select_dtypes(include=['category']).columns
        if pandas_categorical is None:  # train dataset
            pandas_categorical = [list(data[col].cat.categories) for col in cat_cols]
        else:
            if len(cat_cols) != len(pandas_categorical):
                raise ValueError('train and valid dataset categorical_feature do not match.')
            for col, category in zip(cat_cols, pandas_categorical):
                if list(data[col].cat.categories) != list(category):
                    data[col] = data[col].cat.set_categories(category)
        if len(cat_cols):  # cat_cols is pandas Index object
            data = data.copy()  # not alter origin DataFrame
            data[cat_cols] = data[cat_cols].apply(lambda x: x.cat.codes).replace({-1: np.nan})
        if categorical_feature is not None:
            if feature_name is None:
                feature_name = list(data.columns)
            if categorical_feature == 'auto':
                categorical_feature = list(cat_cols)
            else:
                categorical_feature = list(categorical_feature) + list(cat_cols)
        if feature_name == 'auto':
            feature_name = list(data.columns)
        data_dtypes = data.dtypes
        if not all(dtype.name in PANDAS_DTYPE_MAPPER for dtype in data_dtypes):
            bad_fields = [data.columns[i] for i, dtype in
                          enumerate(data_dtypes) if dtype.name not in PANDAS_DTYPE_MAPPER]

            msg = """DataFrame.dtypes for data must be int, float or bool. Did not expect the data types in fields """
            raise ValueError(msg + ', '.join(bad_fields))
        data = data.values.astype('float')
    else:
        if feature_name == 'auto':
            feature_name = None
        if categorical_feature == 'auto':
            categorical_feature = None
    return data, feature_name, categorical_feature, pandas_categorical


def _label_from_pandas(label):
    if isinstance(label, DataFrame):
        if len(label.columns) > 1:
            raise ValueError('DataFrame for label cannot have multiple columns')
        label_dtypes = label.dtypes
        if not all(dtype.name in PANDAS_DTYPE_MAPPER for dtype in label_dtypes):
            raise ValueError('DataFrame.dtypes for label must be int, float or bool')
        label = label.values.astype('float')
    return label


def _save_pandas_categorical(file_name, pandas_categorical):
    with open(file_name, 'a') as f:
        f.write('\npandas_categorical:' + json.dumps(pandas_categorical, default=json_default_with_numpy) + '\n')


def _load_pandas_categorical(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
        last_line = lines[-1]
        if last_line.strip() == "":
            last_line = lines[-2]
        if last_line.startswith('pandas_categorical:'):
            return json.loads(last_line[len('pandas_categorical:'):])
    return None


class _InnerPredictor(object):
    """
    A _InnerPredictor of LightGBM.
    Only used for prediction, usually used for continued-train
    Note: Can convert from Booster, but cannot convert to Booster
    """
    def __init__(self, model_file=None, booster_handle=None, pred_parameter=None):
        """Initialize the _InnerPredictor. Not expose to user

        Parameters
        ----------
        model_file : string
            Path to the model file.
        booster_handle : Handle of Booster
            use handle to init
        pred_parameter: dict
            Other parameters for the prediciton
        """
        self.handle = ctypes.c_void_p()
        self.__is_manage_handle = True
        if model_file is not None:
            """Prediction task"""
            out_num_iterations = ctypes.c_int(0)
            _safe_call(_LIB.LGBM_BoosterCreateFromModelfile(
                c_str(model_file),
                ctypes.byref(out_num_iterations),
                ctypes.byref(self.handle)))
            out_num_class = ctypes.c_int(0)
            _safe_call(_LIB.LGBM_BoosterGetNumClasses(
                self.handle,
                ctypes.byref(out_num_class)))
            self.num_class = out_num_class.value
            self.num_total_iteration = out_num_iterations.value
            self.pandas_categorical = _load_pandas_categorical(model_file)
        elif booster_handle is not None:
            self.__is_manage_handle = False
            self.handle = booster_handle
            out_num_class = ctypes.c_int(0)
            _safe_call(_LIB.LGBM_BoosterGetNumClasses(
                self.handle,
                ctypes.byref(out_num_class)))
            self.num_class = out_num_class.value
            out_num_iterations = ctypes.c_int(0)
            _safe_call(_LIB.LGBM_BoosterGetCurrentIteration(
                self.handle,
                ctypes.byref(out_num_iterations)))
            self.num_total_iteration = out_num_iterations.value
            self.pandas_categorical = None
        else:
            raise TypeError('Need Model file or Booster handle to create a predictor')

        pred_parameter = {} if pred_parameter is None else pred_parameter
        self.pred_parameter = param_dict_to_str(pred_parameter)

    def __del__(self):
        try:
            if self.__is_manage_handle:
                _safe_call(_LIB.LGBM_BoosterFree(self.handle))
        except AttributeError:
            pass

    def __getstate__(self):
        this = self.__dict__.copy()
        this.pop('handle', None)
        return this

    def predict(self, data, num_iteration=-1,
                raw_score=False, pred_leaf=False, pred_contrib=False, data_has_header=False,
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
        pred_contrib : bool
            True for predict feature contributions
        data_has_header : bool
            Used for txt data, True if txt data has header
        is_reshape : bool
            Reshape to (nrow, ncol) if true

        Returns
        -------
        Prediction result
        """
        if isinstance(data, Dataset):
            raise TypeError("Cannot use Dataset instance for prediction, please use raw data instead")
        data = _data_from_pandas(data, None, None, self.pandas_categorical)[0]
        predict_type = C_API_PREDICT_NORMAL
        if raw_score:
            predict_type = C_API_PREDICT_RAW_SCORE
        if pred_leaf:
            predict_type = C_API_PREDICT_LEAF_INDEX
        if pred_contrib:
            predict_type = C_API_PREDICT_CONTRIB
        int_data_has_header = 1 if data_has_header else 0
        if num_iteration > self.num_total_iteration:
            num_iteration = self.num_total_iteration

        if isinstance(data, string_type):
            with _temp_file() as f:
                _safe_call(_LIB.LGBM_BoosterPredictForFile(
                    self.handle,
                    c_str(data),
                    ctypes.c_int(int_data_has_header),
                    ctypes.c_int(predict_type),
                    ctypes.c_int(num_iteration),
                    c_str(self.pred_parameter),
                    c_str(f.name)))
                lines = f.readlines()
                nrow = len(lines)
                preds = [float(token) for line in lines for token in line.split('\t')]
                preds = np.array(preds, dtype=np.float64, copy=False)
        elif isinstance(data, scipy.sparse.csr_matrix):
            preds, nrow = self.__pred_for_csr(data, num_iteration,
                                              predict_type)
        elif isinstance(data, scipy.sparse.csc_matrix):
            preds, nrow = self.__pred_for_csc(data, num_iteration,
                                              predict_type)
        elif isinstance(data, np.ndarray):
            preds, nrow = self.__pred_for_np2d(data, num_iteration,
                                               predict_type)
        elif isinstance(data, list):
            try:
                data = np.array(data)
            except BaseException:
                raise ValueError('Cannot convert data list to numpy array.')
            preds, nrow = self.__pred_for_np2d(data, num_iteration,
                                               predict_type)
        else:
            try:
                warnings.warn('Converting data to scipy sparse matrix.')
                csr = scipy.sparse.csr_matrix(data)
            except BaseException:
                raise TypeError('Cannot predict data for type {}'.format(type(data).__name__))
            preds, nrow = self.__pred_for_csr(csr, num_iteration,
                                              predict_type)
        if pred_leaf:
            preds = preds.astype(np.int32)
        if is_reshape and preds.size != nrow:
            if preds.size % nrow == 0:
                preds = preds.reshape(nrow, -1)
            else:
                raise ValueError('Length of predict result (%d) cannot be divide nrow (%d)'
                                 % (preds.size, nrow))
        return preds

    def __get_num_preds(self, num_iteration, nrow, predict_type):
        """
        Get size of prediction result
        """
        n_preds = ctypes.c_int64(0)
        _safe_call(_LIB.LGBM_BoosterCalcNumPredict(
            self.handle,
            ctypes.c_int(nrow),
            ctypes.c_int(predict_type),
            ctypes.c_int(num_iteration),
            ctypes.byref(n_preds)))
        return n_preds.value

    def __pred_for_np2d(self, mat, num_iteration, predict_type):
        """
        Predict for a 2-D numpy matrix.
        """
        if len(mat.shape) != 2:
            raise ValueError('Input numpy.ndarray or list must be 2 dimensional')

        if mat.dtype == np.float32 or mat.dtype == np.float64:
            data = np.array(mat.reshape(mat.size), dtype=mat.dtype, copy=False)
        else:
            """change non-float data to float data, need to copy"""
            data = np.array(mat.reshape(mat.size), dtype=np.float32)
        ptr_data, type_ptr_data, _ = c_float_array(data)
        n_preds = self.__get_num_preds(num_iteration, mat.shape[0],
                                       predict_type)
        preds = np.zeros(n_preds, dtype=np.float64)
        out_num_preds = ctypes.c_int64(0)
        _safe_call(_LIB.LGBM_BoosterPredictForMat(
            self.handle,
            ptr_data,
            ctypes.c_int(type_ptr_data),
            ctypes.c_int(mat.shape[0]),
            ctypes.c_int(mat.shape[1]),
            ctypes.c_int(C_API_IS_ROW_MAJOR),
            ctypes.c_int(predict_type),
            ctypes.c_int(num_iteration),
            c_str(self.pred_parameter),
            ctypes.byref(out_num_preds),
            preds.ctypes.data_as(ctypes.POINTER(ctypes.c_double))))
        if n_preds != out_num_preds.value:
            raise ValueError("Wrong length for predict results")
        return preds, mat.shape[0]

    def __pred_for_csr(self, csr, num_iteration, predict_type):
        """
        Predict for a csr data
        """
        nrow = len(csr.indptr) - 1
        n_preds = self.__get_num_preds(num_iteration, nrow, predict_type)
        preds = np.zeros(n_preds, dtype=np.float64)
        out_num_preds = ctypes.c_int64(0)

        ptr_indptr, type_ptr_indptr, __ = c_int_array(csr.indptr)
        ptr_data, type_ptr_data, _ = c_float_array(csr.data)

        _safe_call(_LIB.LGBM_BoosterPredictForCSR(
            self.handle,
            ptr_indptr,
            ctypes.c_int32(type_ptr_indptr),
            csr.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ptr_data,
            ctypes.c_int(type_ptr_data),
            ctypes.c_int64(len(csr.indptr)),
            ctypes.c_int64(len(csr.data)),
            ctypes.c_int64(csr.shape[1]),
            ctypes.c_int(predict_type),
            ctypes.c_int(num_iteration),
            c_str(self.pred_parameter),
            ctypes.byref(out_num_preds),
            preds.ctypes.data_as(ctypes.POINTER(ctypes.c_double))))
        if n_preds != out_num_preds.value:
            raise ValueError("Wrong length for predict results")
        return preds, nrow

    def __pred_for_csc(self, csc, num_iteration, predict_type):
        """
        Predict for a csc data
        """
        nrow = csc.shape[0]
        n_preds = self.__get_num_preds(num_iteration, nrow, predict_type)
        preds = np.zeros(n_preds, dtype=np.float64)
        out_num_preds = ctypes.c_int64(0)

        ptr_indptr, type_ptr_indptr, __ = c_int_array(csc.indptr)
        ptr_data, type_ptr_data, _ = c_float_array(csc.data)

        _safe_call(_LIB.LGBM_BoosterPredictForCSC(
            self.handle,
            ptr_indptr,
            ctypes.c_int32(type_ptr_indptr),
            csc.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ptr_data,
            ctypes.c_int(type_ptr_data),
            ctypes.c_int64(len(csc.indptr)),
            ctypes.c_int64(len(csc.data)),
            ctypes.c_int64(csc.shape[0]),
            ctypes.c_int(predict_type),
            ctypes.c_int(num_iteration),
            c_str(self.pred_parameter),
            ctypes.byref(out_num_preds),
            preds.ctypes.data_as(ctypes.POINTER(ctypes.c_double))))
        if n_preds != out_num_preds.value:
            raise ValueError("Wrong length for predict results")
        return preds, nrow


class Dataset(object):
    """Dataset in LightGBM."""
    def __init__(self, data, label=None, reference=None,
                 weight=None, group=None, init_score=None, silent=False,
                 feature_name='auto', categorical_feature='auto', params=None,
                 free_raw_data=True):
        """Constract Dataset.

        Parameters
        ----------
        data : string, numpy array or scipy.sparse
            Data source of Dataset.
            If string, it represents the path to txt file.
        label : list, numpy 1-D array or None, optional (default=None)
            Label of the data.
        reference : Dataset or None, optional (default=None)
            If this is Dataset for validation, training data should be used as reference.
        weight : list, numpy 1-D array or None, optional (default=None)
            Weight for each instance.
        group : list, numpy 1-D array or None, optional (default=None)
            Group/query size for Dataset.
        init_score : list, numpy 1-D array or None, optional (default=None)
            Init score for Dataset.
        silent : bool, optional (default=False)
            Whether to print messages during construction.
        feature_name : list of strings or 'auto', optional (default="auto")
            Feature names.
            If 'auto' and data is pandas DataFrame, data columns names are used.
        categorical_feature : list of strings or int, or 'auto', optional (default="auto")
            Categorical features.
            If list of int, interpreted as indices.
            If list of strings, interpreted as feature names (need to specify ``feature_name`` as well).
            If 'auto' and data is pandas DataFrame, pandas categorical columns are used.
        params: dict or None, optional (default=None)
            Other parameters.
        free_raw_data: bool, optional (default=True)
            If True, raw data is freed after constructing inner Dataset.
        """
        self.handle = None
        self.data = data
        self.label = label
        self.reference = reference
        self.weight = weight
        self.group = group
        self.init_score = init_score
        self.silent = silent
        self.feature_name = feature_name
        self.categorical_feature = categorical_feature
        self.params = copy.deepcopy(params)
        self.free_raw_data = free_raw_data
        self.used_indices = None
        self._predictor = None
        self.pandas_categorical = None
        self.params_back_up = None

    def __del__(self):
        try:
            self._free_handle()
        except AttributeError:
            pass

    def _free_handle(self):
        if self.handle is not None:
            _safe_call(_LIB.LGBM_DatasetFree(self.handle))
            self.handle = None

    def _lazy_init(self, data, label=None, reference=None,
                   weight=None, group=None, init_score=None, predictor=None,
                   silent=False, feature_name='auto',
                   categorical_feature='auto', params=None):
        if data is None:
            self.handle = None
            return
        if reference is not None:
            self.pandas_categorical = reference.pandas_categorical
            categorical_feature = reference.categorical_feature
        data, feature_name, categorical_feature, self.pandas_categorical = _data_from_pandas(data, feature_name, categorical_feature, self.pandas_categorical)
        label = _label_from_pandas(label)
        self.data_has_header = False
        # process for args
        params = {} if params is None else params
        args_names = getattr(self.__class__, '_lazy_init').__code__.co_varnames[:getattr(self.__class__, '_lazy_init').__code__.co_argcount]
        for key, _ in params.items():
            if key in args_names:
                warnings.warn('{0} keyword has been found in `params` and will be ignored. '
                              'Please use {0} argument of the Dataset constructor to pass this parameter.'.format(key))
        self.predictor = predictor
        if "verbosity" in params:
            params.setdefault("verbose", params.pop("verbosity"))
        if silent:
            params["verbose"] = 0
        elif "verbose" not in params:
            params["verbose"] = 1
        # get categorical features
        if categorical_feature is not None:
            categorical_indices = set()
            feature_dict = {}
            if feature_name is not None:
                feature_dict = {name: i for i, name in enumerate(feature_name)}
            for name in categorical_feature:
                if isinstance(name, string_type) and name in feature_dict:
                    categorical_indices.add(feature_dict[name])
                elif isinstance(name, integer_types):
                    categorical_indices.add(name)
                else:
                    raise TypeError("Wrong type({}) or unknown name({}) in categorical_feature"
                                    .format(type(name).__name__, name))
            if categorical_indices:
                if "categorical_feature" in params or "categorical_column" in params:
                    warnings.warn('categorical_feature in param dict is overridden.')
                    params.pop("categorical_feature", None)
                    params.pop("categorical_column", None)
                params['categorical_column'] = sorted(categorical_indices)

        params_str = param_dict_to_str(params)
        # process for reference dataset
        ref_dataset = None
        if isinstance(reference, Dataset):
            ref_dataset = reference.construct().handle
        elif reference is not None:
            raise TypeError('Reference dataset should be None or dataset instance')
        # start construct data
        if isinstance(data, string_type):
            # check data has header or not
            if str(params.get("has_header", "")).lower() == "true" \
                    or str(params.get("header", "")).lower() == "true":
                self.data_has_header = True
            self.handle = ctypes.c_void_p()
            _safe_call(_LIB.LGBM_DatasetCreateFromFile(
                c_str(data),
                c_str(params_str),
                ref_dataset,
                ctypes.byref(self.handle)))
        elif isinstance(data, scipy.sparse.csr_matrix):
            self.__init_from_csr(data, params_str, ref_dataset)
        elif isinstance(data, scipy.sparse.csc_matrix):
            self.__init_from_csc(data, params_str, ref_dataset)
        elif isinstance(data, np.ndarray):
            self.__init_from_np2d(data, params_str, ref_dataset)
        else:
            try:
                csr = scipy.sparse.csr_matrix(data)
                self.__init_from_csr(csr, params_str, ref_dataset)
            except BaseException:
                raise TypeError('Cannot initialize Dataset from {}'.format(type(data).__name__))
        if label is not None:
            self.set_label(label)
        if self.get_label() is None:
            raise ValueError("Label should not be None")
        if weight is not None:
            self.set_weight(weight)
        if group is not None:
            self.set_group(group)
        # load init score
        if init_score is not None:
            self.set_init_score(init_score)
            if self.predictor is not None:
                warnings.warn("The prediction of init_model will be overridden by init_score.")
        elif isinstance(self.predictor, _InnerPredictor):
            init_score = self.predictor.predict(data,
                                                raw_score=True,
                                                data_has_header=self.data_has_header,
                                                is_reshape=False)
            if self.predictor.num_class > 1:
                # need re group init score
                new_init_score = np.zeros(init_score.size, dtype=np.float32)
                num_data = self.num_data()
                for i in range_(num_data):
                    for j in range_(self.predictor.num_class):
                        new_init_score[j * num_data + i] = init_score[i * self.predictor.num_class + j]
                init_score = new_init_score
            self.set_init_score(init_score)
        elif self.predictor is not None:
            raise TypeError('wrong predictor type {}'.format(type(self.predictor).__name__))
        # set feature names
        self.set_feature_name(feature_name)

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
            # change non-float data to float data, need to copy
            data = np.array(mat.reshape(mat.size), dtype=np.float32)

        ptr_data, type_ptr_data, _ = c_float_array(data)
        _safe_call(_LIB.LGBM_DatasetCreateFromMat(
            ptr_data,
            ctypes.c_int(type_ptr_data),
            ctypes.c_int(mat.shape[0]),
            ctypes.c_int(mat.shape[1]),
            ctypes.c_int(C_API_IS_ROW_MAJOR),
            c_str(params_str),
            ref_dataset,
            ctypes.byref(self.handle)))

    def __init_from_csr(self, csr, params_str, ref_dataset):
        """
        Initialize data from a CSR matrix.
        """
        if len(csr.indices) != len(csr.data):
            raise ValueError('Length mismatch: {} vs {}'.format(len(csr.indices), len(csr.data)))
        self.handle = ctypes.c_void_p()

        ptr_indptr, type_ptr_indptr, __ = c_int_array(csr.indptr)
        ptr_data, type_ptr_data, _ = c_float_array(csr.data)

        _safe_call(_LIB.LGBM_DatasetCreateFromCSR(
            ptr_indptr,
            ctypes.c_int(type_ptr_indptr),
            csr.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ptr_data,
            ctypes.c_int(type_ptr_data),
            ctypes.c_int64(len(csr.indptr)),
            ctypes.c_int64(len(csr.data)),
            ctypes.c_int64(csr.shape[1]),
            c_str(params_str),
            ref_dataset,
            ctypes.byref(self.handle)))

    def __init_from_csc(self, csc, params_str, ref_dataset):
        """
        Initialize data from a csc matrix.
        """
        if len(csc.indices) != len(csc.data):
            raise ValueError('Length mismatch: {} vs {}'.format(len(csc.indices), len(csc.data)))
        self.handle = ctypes.c_void_p()

        ptr_indptr, type_ptr_indptr, __ = c_int_array(csc.indptr)
        ptr_data, type_ptr_data, _ = c_float_array(csc.data)

        _safe_call(_LIB.LGBM_DatasetCreateFromCSC(
            ptr_indptr,
            ctypes.c_int(type_ptr_indptr),
            csc.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ptr_data,
            ctypes.c_int(type_ptr_data),
            ctypes.c_int64(len(csc.indptr)),
            ctypes.c_int64(len(csc.data)),
            ctypes.c_int64(csc.shape[0]),
            c_str(params_str),
            ref_dataset,
            ctypes.byref(self.handle)))

    def construct(self):
        """Lazy init.

        Returns
        -------
        self : Dataset
            Returns self.
        """
        if self.handle is None:
            if self.reference is not None:
                if self.used_indices is None:
                    # create valid
                    self._lazy_init(self.data, label=self.label, reference=self.reference,
                                    weight=self.weight, group=self.group, init_score=self.init_score, predictor=self._predictor,
                                    silent=self.silent, feature_name=self.feature_name, params=self.params)
                else:
                    # construct subset
                    used_indices = list_to_1d_numpy(self.used_indices, np.int32, name='used_indices')
                    assert used_indices.flags.c_contiguous
                    self.handle = ctypes.c_void_p()
                    params_str = param_dict_to_str(self.params)
                    _safe_call(_LIB.LGBM_DatasetGetSubset(
                        self.reference.construct().handle,
                        used_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                        ctypes.c_int(used_indices.shape[0]),
                        c_str(params_str),
                        ctypes.byref(self.handle)))
                    if self.get_label() is None:
                        raise ValueError("Label should not be None.")
            else:
                # create train
                self._lazy_init(self.data, label=self.label,
                                weight=self.weight, group=self.group, init_score=self.init_score,
                                predictor=self._predictor, silent=self.silent, feature_name=self.feature_name,
                                categorical_feature=self.categorical_feature, params=self.params)
            if self.free_raw_data:
                self.data = None
        return self

    def create_valid(self, data, label=None, weight=None, group=None,
                     init_score=None, silent=False, params=None):
        """Create validation data align with current Dataset.

        Parameters
        ----------
        data : string, numpy array or scipy.sparse
            Data source of Dataset.
            If string, it represents the path to txt file.
        label : list or numpy 1-D array, optional (default=None)
            Label of the training data.
        weight : list, numpy 1-D array or None, optional (default=None)
            Weight for each instance.
        group : list, numpy 1-D array or None, optional (default=None)
            Group/query size for Dataset.
        init_score : list, numpy 1-D array or None, optional (default=None)
            Init score for Dataset.
        silent : bool, optional (default=False)
            Whether to print messages during construction.
        params: dict or None, optional (default=None)
            Other parameters.

        Returns
        -------
        self : Dataset
            Returns self.
        """
        ret = Dataset(data, label=label, reference=self,
                      weight=weight, group=group, init_score=init_score,
                      silent=silent, params=params, free_raw_data=self.free_raw_data)
        ret._predictor = self._predictor
        ret.pandas_categorical = self.pandas_categorical
        return ret

    def subset(self, used_indices, params=None):
        """Get subset of current Dataset.

        Parameters
        ----------
        used_indices : list of int
            Indices used to create the subset.
        params: dict or None, optional (default=None)
            Other parameters.

        Returns
        -------
        subset : Dataset
            Subset of the current Dataset.
        """
        if params is None:
            params = self.params
        ret = Dataset(None, reference=self, feature_name=self.feature_name,
                      categorical_feature=self.categorical_feature, params=params)
        ret._predictor = self._predictor
        ret.pandas_categorical = self.pandas_categorical
        ret.used_indices = used_indices
        return ret

    def save_binary(self, filename):
        """Save Dataset to binary file.

        Parameters
        ----------
        filename : string
            Name of the output file.
        """
        _safe_call(_LIB.LGBM_DatasetSaveBinary(
            self.construct().handle,
            c_str(filename)))

    def _update_params(self, params):
        if not self.params:
            self.params = params
        else:
            self.params_back_up = copy.deepcopy(self.params)
            self.params.update(params)

    def _reverse_update_params(self):
        self.params = copy.deepcopy(self.params_back_up)
        self.params_back_up = None

    def set_field(self, field_name, data):
        """Set property into the Dataset.

        Parameters
        ----------
        field_name: string
            The field name of the information.
        data: list, numpy array or None
            The array of data to be set.
        """
        if self.handle is None:
            raise Exception("Cannot set %s before construct dataset" % field_name)
        if data is None:
            # set to None
            _safe_call(_LIB.LGBM_DatasetSetField(
                self.handle,
                c_str(field_name),
                None,
                ctypes.c_int(0),
                ctypes.c_int(FIELD_TYPE_MAPPER[field_name])))
            return
        dtype = np.float32
        if field_name == 'group':
            dtype = np.int32
        elif field_name == 'init_score':
            dtype = np.float64
        data = list_to_1d_numpy(data, dtype, name=field_name)
        if data.dtype == np.float32 or data.dtype == np.float64:
            ptr_data, type_data, _ = c_float_array(data)
        elif data.dtype == np.int32:
            ptr_data, type_data, _ = c_int_array(data)
        else:
            raise TypeError("Excepted np.float32/64 or np.int32, meet type({})".format(data.dtype))
        if type_data != FIELD_TYPE_MAPPER[field_name]:
            raise TypeError("Input type error for set_field")
        _safe_call(_LIB.LGBM_DatasetSetField(
            self.handle,
            c_str(field_name),
            ptr_data,
            ctypes.c_int(len(data)),
            ctypes.c_int(type_data)))

    def get_field(self, field_name):
        """Get property from the Dataset.

        Parameters
        ----------
        field_name: string
            The field name of the information.

        Returns
        -------
        info : numpy array
            A numpy array with information from the Dataset.
        """
        if self.handle is None:
            raise Exception("Cannot get %s before construct Dataset" % field_name)
        tmp_out_len = ctypes.c_int()
        out_type = ctypes.c_int()
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
        elif out_type.value == C_API_DTYPE_FLOAT64:
            return cfloat64_array_to_numpy(ctypes.cast(ret, ctypes.POINTER(ctypes.c_double)), tmp_out_len.value)
        else:
            raise TypeError("Unknown type")

    def set_categorical_feature(self, categorical_feature):
        """Set categorical features.

        Parameters
        ----------
        categorical_feature : list of int or strings
            Names or indices of categorical features.
        """
        if self.categorical_feature == categorical_feature:
            return
        if self.data is not None:
            if self.categorical_feature is None:
                self.categorical_feature = categorical_feature
                self._free_handle()
            elif categorical_feature == 'auto':
                warnings.warn('Using categorical_feature in Dataset.')
            else:
                warnings.warn('categorical_feature in Dataset is overridden. New categorical_feature is {}'.format(sorted(list(categorical_feature))))
                self.categorical_feature = categorical_feature
                self._free_handle()
        else:
            raise LightGBMError("Cannot set categorical feature after freed raw data, set free_raw_data=False when construct Dataset to avoid this.")

    def _set_predictor(self, predictor):
        """
        Set predictor for continued training, not recommand for user to call this function.
        Please set init_model in engine.train or engine.cv
        """
        if predictor is self._predictor:
            return
        if self.data is not None:
            self._predictor = predictor
            self._free_handle()
        else:
            raise LightGBMError("Cannot set predictor after freed raw data, set free_raw_data=False when construct Dataset to avoid this.")

    def set_reference(self, reference):
        """Set reference Dataset.

        Parameters
        ----------
        reference : Dataset
            Reference that is used as a template to consturct the current Dataset.
        """
        self.set_categorical_feature(reference.categorical_feature)
        self.set_feature_name(reference.feature_name)
        self._set_predictor(reference._predictor)
        # we're done if self and reference share a common upstrem reference
        if self.get_ref_chain().intersection(reference.get_ref_chain()):
            return
        if self.data is not None:
            self.reference = reference
            self._free_handle()
        else:
            raise LightGBMError("Cannot set reference after freed raw data, set free_raw_data=False when construct Dataset to avoid this.")

    def set_feature_name(self, feature_name):
        """Set feature name.

        Parameters
        ----------
        feature_name : list of strings
            Feature names.
        """
        if feature_name != 'auto':
            self.feature_name = feature_name
        if self.handle is not None and feature_name is not None and feature_name != 'auto':
            if len(feature_name) != self.num_feature():
                raise ValueError("Length of feature_name({}) and num_feature({}) don't match".format(len(feature_name), self.num_feature()))
            c_feature_name = [c_str(name) for name in feature_name]
            _safe_call(_LIB.LGBM_DatasetSetFeatureNames(
                self.handle,
                c_array(ctypes.c_char_p, c_feature_name),
                ctypes.c_int(len(feature_name))))

    def set_label(self, label):
        """Set label of Dataset

        Parameters
        ----------
        label: list, numpy array or None
            The label information to be set into Dataset.
        """
        self.label = label
        if self.handle is not None:
            label = list_to_1d_numpy(label, name='label')
            self.set_field('label', label)

    def set_weight(self, weight):
        """Set weight of each instance.

        Parameters
        ----------
        weight : list, numpy array or None
            Weight to be set for each data point.
        """
        if weight is not None and np.all(weight == 1):
            weight = None
        self.weight = weight
        if self.handle is not None and weight is not None:
            weight = list_to_1d_numpy(weight, name='weight')
            self.set_field('weight', weight)

    def set_init_score(self, init_score):
        """Set init score of Booster to start from.

        Parameters
        ----------
        init_score : list, numpy array or None
            Init score for Booster.
        """
        self.init_score = init_score
        if self.handle is not None and init_score is not None:
            init_score = list_to_1d_numpy(init_score, np.float64, name='init_score')
            self.set_field('init_score', init_score)

    def set_group(self, group):
        """Set group size of Dataset (used for ranking).

        Parameters
        ----------
        group : list, numpy array or None
            Group size of each group.
        """
        self.group = group
        if self.handle is not None and group is not None:
            group = list_to_1d_numpy(group, np.int32, name='group')
            self.set_field('group', group)

    def get_label(self):
        """Get the label of the Dataset.

        Returns
        -------
        label : numpy array
            The label information from the Dataset.
        """
        if self.label is None:
            self.label = self.get_field('label')
        return self.label

    def get_weight(self):
        """Get the weight of the Dataset.

        Returns
        -------
        weight : numpy array
            Weight for each data point from the Dataset.
        """
        if self.weight is None:
            self.weight = self.get_field('weight')
        return self.weight

    def get_init_score(self):
        """Get the initial score of the Dataset.

        Returns
        -------
        init_score : numpy array
            Init score of Booster.
        """
        if self.init_score is None:
            self.init_score = self.get_field('init_score')
        return self.init_score

    def get_group(self):
        """Get the group of the Dataset.

        Returns
        -------
        group : numpy array
            Group size of each group.
        """
        if self.group is None:
            self.group = self.get_field('group')
            if self.group is not None:
                # group data from LightGBM is boundaries data, need to convert to group size
                new_group = []
                for i in range_(len(self.group) - 1):
                    new_group.append(self.group[i + 1] - self.group[i])
                self.group = new_group
        return self.group

    def num_data(self):
        """Get the number of rows in the Dataset.

        Returns
        -------
        number_of_rows : int
            The number of rows in the Dataset.
        """
        if self.handle is not None:
            ret = ctypes.c_int()
            _safe_call(_LIB.LGBM_DatasetGetNumData(self.handle,
                                                   ctypes.byref(ret)))
            return ret.value
        else:
            raise LightGBMError("Cannot get num_data before construct dataset")

    def num_feature(self):
        """Get the number of columns (features) in the Dataset.

        Returns
        -------
        number_of_columns : int
            The number of columns (features) in the Dataset.
        """
        if self.handle is not None:
            ret = ctypes.c_int()
            _safe_call(_LIB.LGBM_DatasetGetNumFeature(self.handle,
                                                      ctypes.byref(ret)))
            return ret.value
        else:
            raise LightGBMError("Cannot get num_feature before construct dataset")

    def get_ref_chain(self, ref_limit=100):
        """Get a chain of Dataset objects, starting with r, then going to r.reference if exists,
        then to r.reference.reference, etc. until we hit ``ref_limit`` or a reference loop.

        Parameters
        ----------
        ref_limit : int, optional (default=100)
            The limit number of references.

        Returns
        -------
        ref_chain : set of Dataset
            Chain of references of the Datasets.
        """
        head = self
        ref_chain = set()
        while len(ref_chain) < ref_limit:
            if isinstance(head, Dataset):
                ref_chain.add(head)
                if (head.reference is not None) and (head.reference not in ref_chain):
                    head = head.reference
                else:
                    break
            else:
                break
        return(ref_chain)


class Booster(object):
    """Booster in LightGBM."""
    def __init__(self, params=None, train_set=None, model_file=None, silent=False):
        """Initialize the Booster.

        Parameters
        ----------
        params: dict or None, optional (default=None)
            Parameters for Booster.
        train_set : Dataset or None, optional (default=None)
            Training dataset.
        model_file : string or None, optional (default=None)
            Path to the model file.
        silent : bool, optional (default=False)
            Whether to print messages during construction.
        """
        self.handle = None
        self.network = False
        self.__need_reload_eval_info = True
        self.__train_data_name = "training"
        self.__attr = {}
        self.__set_objective_to_none = False
        self.best_iteration = -1
        self.best_score = {}
        params = {} if params is None else params
        if "verbosity" in params:
            params.setdefault("verbose", params.pop("verbosity"))
        if silent:
            params["verbose"] = 0
        elif "verbose" not in params:
            params["verbose"] = 1
        if train_set is not None:
            # Training task
            if not isinstance(train_set, Dataset):
                raise TypeError('Training data should be Dataset instance, met {}'.format(type(train_set).__name__))
            params_str = param_dict_to_str(params)
            # construct booster object
            self.handle = ctypes.c_void_p()
            _safe_call(_LIB.LGBM_BoosterCreate(
                train_set.construct().handle,
                c_str(params_str),
                ctypes.byref(self.handle)))
            # save reference to data
            self.train_set = train_set
            self.valid_sets = []
            self.name_valid_sets = []
            self.__num_dataset = 1
            self.__init_predictor = train_set._predictor
            if self.__init_predictor is not None:
                _safe_call(_LIB.LGBM_BoosterMerge(
                    self.handle,
                    self.__init_predictor.handle))
            out_num_class = ctypes.c_int(0)
            _safe_call(_LIB.LGBM_BoosterGetNumClasses(
                self.handle,
                ctypes.byref(out_num_class)))
            self.__num_class = out_num_class.value
            # buffer for inner predict
            self.__inner_predict_buffer = [None]
            self.__is_predicted_cur_iter = [False]
            self.__get_eval_info()
            self.pandas_categorical = train_set.pandas_categorical
            # set network if necessary
            if "machines" in params:
                machines = params["machines"]
                if isinstance(machines, string_type):
                    num_machines = len(machines.split(','))
                elif isinstance(machines, (list, set)):
                    num_machines = len(machines)
                    machines = ','.join(machines)
                else:
                    raise ValueError("Invalid machines in params.")
                self.set_network(machines,
                                 local_listen_port=params.get("local_listen_port", 12400),
                                 listen_time_out=params.get("listen_time_out", 120),
                                 num_machines=params.get("num_machines", num_machines))
        elif model_file is not None:
            # Prediction task
            out_num_iterations = ctypes.c_int(0)
            self.handle = ctypes.c_void_p()
            _safe_call(_LIB.LGBM_BoosterCreateFromModelfile(
                c_str(model_file),
                ctypes.byref(out_num_iterations),
                ctypes.byref(self.handle)))
            out_num_class = ctypes.c_int(0)
            _safe_call(_LIB.LGBM_BoosterGetNumClasses(
                self.handle,
                ctypes.byref(out_num_class)))
            self.__num_class = out_num_class.value
            self.pandas_categorical = _load_pandas_categorical(model_file)
        elif 'model_str' in params:
            self._load_model_from_string(params['model_str'])
        else:
            raise TypeError('Need at least one training dataset or model file to create booster instance')

    def __del__(self):
        try:
            if self.network:
                self.free_network()
        except AttributeError:
            pass
        try:
            if self.handle is not None:
                _safe_call(_LIB.LGBM_BoosterFree(self.handle))
        except AttributeError:
            pass

    def __copy__(self):
        return self.__deepcopy__(None)

    def __deepcopy__(self, _):
        model_str = self._save_model_to_string()
        booster = Booster({'model_str': model_str})
        booster.pandas_categorical = self.pandas_categorical
        return booster

    def __getstate__(self):
        this = self.__dict__.copy()
        handle = this['handle']
        this.pop('train_set', None)
        this.pop('valid_sets', None)
        if handle is not None:
            this["handle"] = self._save_model_to_string()
        return this

    def __setstate__(self, state):
        model_str = state.get('handle', None)
        if model_str is not None:
            handle = ctypes.c_void_p()
            out_num_iterations = ctypes.c_int(0)
            _safe_call(_LIB.LGBM_BoosterLoadModelFromString(
                c_str(model_str),
                ctypes.byref(out_num_iterations),
                ctypes.byref(handle)))
            state['handle'] = handle
        self.__dict__.update(state)

    def free_dataset(self):
        """Free Booster's Datasets."""
        self.__dict__.pop('train_set', None)
        self.__dict__.pop('valid_sets', None)
        self.__num_dataset = 0

    def _free_buffer(self):
        self.__inner_predict_buffer = []
        self.__is_predicted_cur_iter = []

    def set_network(self, machines, local_listen_port=12400,
                    listen_time_out=120, num_machines=1):
        """Set the network configuration.

        Parameters
        ----------
        machines: list, set or string
            Names of machines.
        local_listen_port: int, optional (default=12400)
            TCP listen port for local machines.
        listen_time_out: int, optional (default=120)
            Socket time-out in minutes.
        num_machines: int, optional (default=1)
            The number of machines for parallel learning application.
        """
        _safe_call(_LIB.LGBM_NetworkInit(c_str(machines),
                                         ctypes.c_int(local_listen_port),
                                         ctypes.c_int(listen_time_out),
                                         ctypes.c_int(num_machines)))
        self.network = True

    def free_network(self):
        """Free network."""
        _safe_call(_LIB.LGBM_NetworkFree())
        self.network = False

    def set_train_data_name(self, name):
        """Set the name to the training Dataset.

        Parameters
        ----------
        name: string
            Name for training Dataset.
        """
        self.__train_data_name = name

    def add_valid(self, data, name):
        """Add validation data.

        Parameters
        ----------
        data : Dataset
            Validation data.
        name : string
            Name of validation data.
        """
        if not isinstance(data, Dataset):
            raise TypeError('Validation data should be Dataset instance, met {}'.format(type(data).__name__))
        if data._predictor is not self.__init_predictor:
            raise LightGBMError("Add validation data failed, you should use same predictor for these data")
        _safe_call(_LIB.LGBM_BoosterAddValidData(
            self.handle,
            data.construct().handle))
        self.valid_sets.append(data)
        self.name_valid_sets.append(name)
        self.__num_dataset += 1
        self.__inner_predict_buffer.append(None)
        self.__is_predicted_cur_iter.append(False)

    def reset_parameter(self, params):
        """Reset parameters of Booster.

        Parameters
        ----------
        params : dict
            New parameters for Booster.
        """
        if 'metric' in params:
            self.__need_reload_eval_info = True
        params_str = param_dict_to_str(params)
        if params_str:
            _safe_call(_LIB.LGBM_BoosterResetParameter(
                self.handle,
                c_str(params_str)))

    def update(self, train_set=None, fobj=None):
        """Update for one iteration.

        Parameters
        ----------
        train_set : Dataset or None, optional (default=None)
            Training data.
            If None, last training data is used.
        fobj : callable or None, optional (default=None)
            Customized objective function.

            For multi-class task, the score is group by class_id first, then group by row_id.
            If you want to get i-th row score in j-th class, the access way is score[j * num_data + i]
            and you should group grad and hess in this way as well.

        Returns
        -------
        is_finished : bool
            Whether the update was successfully finished.
        """

        # need reset training data
        if train_set is not None and train_set is not self.train_set:
            if not isinstance(train_set, Dataset):
                raise TypeError('Training data should be Dataset instance, met {}'.format(type(train_set).__name__))
            if train_set._predictor is not self.__init_predictor:
                raise LightGBMError("Replace training data failed, you should use same predictor for these data")
            self.train_set = train_set
            _safe_call(_LIB.LGBM_BoosterResetTrainingData(
                self.handle,
                self.train_set.construct().handle))
            self.__inner_predict_buffer[0] = None
        is_finished = ctypes.c_int(0)
        if fobj is None:
            if self.__set_objective_to_none:
                raise ValueError('Cannot update due to null objective function.')
            _safe_call(_LIB.LGBM_BoosterUpdateOneIter(
                self.handle,
                ctypes.byref(is_finished)))
            self.__is_predicted_cur_iter = [False for _ in range_(self.__num_dataset)]
            return is_finished.value == 1
        else:
            if not self.__set_objective_to_none:
                self.reset_parameter({"objective": "none"})
                self.__set_objective_to_none = True
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
        grad = list_to_1d_numpy(grad, name='gradient')
        hess = list_to_1d_numpy(hess, name='hessian')
        assert grad.flags.c_contiguous
        assert hess.flags.c_contiguous
        if len(grad) != len(hess):
            raise ValueError("Lengths of gradient({}) and hessian({}) don't match".format(len(grad), len(hess)))
        is_finished = ctypes.c_int(0)
        _safe_call(_LIB.LGBM_BoosterUpdateOneIterCustom(
            self.handle,
            grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            hess.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(is_finished)))
        self.__is_predicted_cur_iter = [False for _ in range_(self.__num_dataset)]
        return is_finished.value == 1

    def rollback_one_iter(self):
        """Rollback one iteration."""
        _safe_call(_LIB.LGBM_BoosterRollbackOneIter(
            self.handle))
        self.__is_predicted_cur_iter = [False for _ in range_(self.__num_dataset)]

    def current_iteration(self):
        """Get the index of the current iteration.

        Returns
        -------
        cur_iter : int
            The index of the current iteration.
        """
        out_cur_iter = ctypes.c_int(0)
        _safe_call(_LIB.LGBM_BoosterGetCurrentIteration(
            self.handle,
            ctypes.byref(out_cur_iter)))
        return out_cur_iter.value

    def eval(self, data, name, feval=None):
        """Evaluate for data.

        Parameters
        ----------
        data : Dataset
            Data for the evaluating.
        name : string
            Name of the data.
        feval : callable or None, optional (default=None)
            Custom evaluation function.

        Returns
        -------
        result: list
            List with evaluation results.
        """
        if not isinstance(data, Dataset):
            raise TypeError("Can only eval for Dataset instance")
        data_idx = -1
        if data is self.train_set:
            data_idx = 0
        else:
            for i in range_(len(self.valid_sets)):
                if data is self.valid_sets[i]:
                    data_idx = i + 1
                    break
        # need to push new valid data
        if data_idx == -1:
            self.add_valid(data, name)
            data_idx = self.__num_dataset - 1

        return self.__inner_eval(name, data_idx, feval)

    def eval_train(self, feval=None):
        """Evaluate for training data.

        Parameters
        ----------
        feval : callable or None, optional (default=None)
            Custom evaluation function.

        Returns
        -------
        result: list
            List with evaluation results.
        """
        return self.__inner_eval(self.__train_data_name, 0, feval)

    def eval_valid(self, feval=None):
        """Evaluate for validation data.

        Parameters
        ----------
        feval : callable or None, optional (default=None)
            Custom evaluation function.

        Returns
        -------
        result: list
            List with evaluation results.
        """
        return [item for i in range_(1, self.__num_dataset)
                for item in self.__inner_eval(self.name_valid_sets[i - 1], i, feval)]

    def save_model(self, filename, num_iteration=-1):
        """Save Booster to file.

        Parameters
        ----------
        filename : string
            Filename to save Booster.
        num_iteration: int, optional (default=-1)
            Index of the iteration that should to saved.
            If <0, the best iteration (if exists) is saved.
        """
        if num_iteration <= 0:
            num_iteration = self.best_iteration
        _safe_call(_LIB.LGBM_BoosterSaveModel(
            self.handle,
            ctypes.c_int(num_iteration),
            c_str(filename)))
        _save_pandas_categorical(filename, self.pandas_categorical)

    def _load_model_from_string(self, model_str, verbose=True):
        """[Private] Load model from string"""
        if self.handle is not None:
            _safe_call(_LIB.LGBM_BoosterFree(self.handle))
        self._free_buffer()
        self.handle = ctypes.c_void_p()
        out_num_iterations = ctypes.c_int(0)
        _safe_call(_LIB.LGBM_BoosterLoadModelFromString(
            c_str(model_str),
            ctypes.byref(out_num_iterations),
            ctypes.byref(self.handle)))
        out_num_class = ctypes.c_int(0)
        _safe_call(_LIB.LGBM_BoosterGetNumClasses(
            self.handle,
            ctypes.byref(out_num_class)))
        if verbose:
            print('Finished loading model, total used %d iterations' % (int(out_num_iterations.value)))
        self.__num_class = out_num_class.value

    def _save_model_to_string(self, num_iteration=-1):
        """[Private] Save model to string"""
        if num_iteration <= 0:
            num_iteration = self.best_iteration
        buffer_len = 1 << 20
        tmp_out_len = ctypes.c_int64(0)
        string_buffer = ctypes.create_string_buffer(buffer_len)
        ptr_string_buffer = ctypes.c_char_p(*[ctypes.addressof(string_buffer)])
        _safe_call(_LIB.LGBM_BoosterSaveModelToString(
            self.handle,
            ctypes.c_int(num_iteration),
            ctypes.c_int64(buffer_len),
            ctypes.byref(tmp_out_len),
            ptr_string_buffer))
        actual_len = tmp_out_len.value
        '''if buffer length is not long enough, re-allocate a buffer'''
        if actual_len > buffer_len:
            string_buffer = ctypes.create_string_buffer(actual_len)
            ptr_string_buffer = ctypes.c_char_p(*[ctypes.addressof(string_buffer)])
            _safe_call(_LIB.LGBM_BoosterSaveModelToString(
                self.handle,
                ctypes.c_int(num_iteration),
                ctypes.c_int64(actual_len),
                ctypes.byref(tmp_out_len),
                ptr_string_buffer))
        return string_buffer.value.decode()

    def dump_model(self, num_iteration=-1):
        """Dump Booster to json format.

        Parameters
        ----------
        num_iteration: int, optional (default=-1)
            Index of the iteration that should to dumped.
            If <0, the best iteration (if exists) is dumped.

        Returns
        -------
        json_repr : dict
            Json format of Booster.
        """
        if num_iteration <= 0:
            num_iteration = self.best_iteration
        buffer_len = 1 << 20
        tmp_out_len = ctypes.c_int64(0)
        string_buffer = ctypes.create_string_buffer(buffer_len)
        ptr_string_buffer = ctypes.c_char_p(*[ctypes.addressof(string_buffer)])
        _safe_call(_LIB.LGBM_BoosterDumpModel(
            self.handle,
            ctypes.c_int(num_iteration),
            ctypes.c_int64(buffer_len),
            ctypes.byref(tmp_out_len),
            ptr_string_buffer))
        actual_len = tmp_out_len.value
        '''if buffer length is not long enough, reallocate a buffer'''
        if actual_len > buffer_len:
            string_buffer = ctypes.create_string_buffer(actual_len)
            ptr_string_buffer = ctypes.c_char_p(*[ctypes.addressof(string_buffer)])
            _safe_call(_LIB.LGBM_BoosterDumpModel(
                self.handle,
                ctypes.c_int(num_iteration),
                ctypes.c_int64(actual_len),
                ctypes.byref(tmp_out_len),
                ptr_string_buffer))
        return json.loads(string_buffer.value.decode())

    def predict(self, data, num_iteration=-1, raw_score=False, pred_leaf=False, pred_contrib=False,
                data_has_header=False, is_reshape=True, pred_parameter=None, **kwargs):
        """Make a prediction.

        Parameters
        ----------
        data : string, numpy array or scipy.sparse
            Data source for prediction.
            If string, it represents the path to txt file.
        num_iteration : int, optional (default=-1)
            Iteration used for prediction.
            If <0, the best iteration (if exists) is used for prediction.
        raw_score : bool, optional (default=False)
            Whether to predict raw scores.
        pred_leaf : bool, optional (default=False)
            Whether to predict leaf index.
        pred_contrib : bool, optional (default=False)
            Whether to predict feature contributions.
        data_has_header : bool, optional (default=False)
            Whether the data has header.
            Used only if data is string.
        is_reshape : bool, optional (default=True)
            If True, result is reshaped to [nrow, ncol].
        pred_parameter : dict or None, optional (default=None)
            Deprecated.
            Other parameters for the prediction.
        **kwargs : other parameters for the prediction

        Returns
        -------
        result : numpy array
            Prediction result.
        """
        if pred_parameter:
            warnings.warn("pred_parameter is deprecated and will be removed in 2.2 version.\n"
                          "Please use kwargs instead.", LGBMDeprecationWarning)
            pred_parameter.update(kwargs)
        else:
            pred_parameter = kwargs
        predictor = self._to_predictor(pred_parameter)
        if num_iteration <= 0:
            num_iteration = self.best_iteration
        return predictor.predict(data, num_iteration, raw_score, pred_leaf, pred_contrib, data_has_header, is_reshape)

    def get_leaf_output(self, tree_id, leaf_id):
        """Get the output of a leaf.

        Parameters
        ----------
        tree_id : int
            The index of the tree.
        leaf_id : int
            The index of the leaf in the tree.

        Returns
        -------
        result : float
            The output of the leaf.
        """
        ret = ctypes.c_double(0)
        _safe_call(_LIB.LGBM_BoosterGetLeafValue(
            self.handle,
            ctypes.c_int(tree_id),
            ctypes.c_int(leaf_id),
            ctypes.byref(ret)))
        return ret.value

    def _to_predictor(self, pred_parameter=None):
        """Convert to predictor"""
        predictor = _InnerPredictor(booster_handle=self.handle, pred_parameter=pred_parameter)
        predictor.pandas_categorical = self.pandas_categorical
        return predictor

    def num_feature(self):
        """Get number of features.

        Returns
        -------
        num_feature : int
            The number of features.
        """
        out_num_feature = ctypes.c_int(0)
        _safe_call(_LIB.LGBM_BoosterGetNumFeature(
            self.handle,
            ctypes.byref(out_num_feature)))
        return out_num_feature.value

    def feature_name(self):
        """Get names of features.

        Returns
        -------
        result : list
            List with names of features.
        """
        num_feature = self.num_feature()
        # Get name of features
        tmp_out_len = ctypes.c_int(0)
        string_buffers = [ctypes.create_string_buffer(255) for i in range_(num_feature)]
        ptr_string_buffers = (ctypes.c_char_p * num_feature)(*map(ctypes.addressof, string_buffers))
        _safe_call(_LIB.LGBM_BoosterGetFeatureNames(
            self.handle,
            ctypes.byref(tmp_out_len),
            ptr_string_buffers))
        if num_feature != tmp_out_len.value:
            raise ValueError("Length of feature names doesn't equal with num_feature")
        return [string_buffers[i].value.decode() for i in range_(num_feature)]

    def feature_importance(self, importance_type='split', iteration=-1):
        """Get feature importances.

        Parameters
        ----------
        importance_type : string, optional (default="split")
            How the importance is calculated.
            If "split", result contains numbers of times the feature is used in a model.
            If "gain", result contains total gains of splits which use the feature.

        Returns
        -------
        result : numpy array
            Array with feature importances.
        """
        if importance_type == "split":
            importance_type_int = 0
        elif importance_type == "gain":
            importance_type_int = 1
        else:
            importance_type_int = -1
        num_feature = self.num_feature()
        result = np.array([0 for _ in range_(num_feature)], dtype=np.float64)
        _safe_call(_LIB.LGBM_BoosterFeatureImportance(
            self.handle,
            ctypes.c_int(iteration),
            ctypes.c_int(importance_type_int),
            result.ctypes.data_as(ctypes.POINTER(ctypes.c_double))))
        if importance_type_int == 0:
            return result.astype(int)
        else:
            return result

    def __inner_eval(self, data_name, data_idx, feval=None):
        """
        Evaulate training or validation data
        """
        if data_idx >= self.__num_dataset:
            raise ValueError("Data_idx should be smaller than number of dataset")
        self.__get_eval_info()
        ret = []
        if self.__num_inner_eval > 0:
            result = np.array([0.0 for _ in range_(self.__num_inner_eval)], dtype=np.float64)
            tmp_out_len = ctypes.c_int(0)
            _safe_call(_LIB.LGBM_BoosterGetEval(
                self.handle,
                ctypes.c_int(data_idx),
                ctypes.byref(tmp_out_len),
                result.ctypes.data_as(ctypes.POINTER(ctypes.c_double))))
            if tmp_out_len.value != self.__num_inner_eval:
                raise ValueError("Wrong length of eval results")
            for i in range_(self.__num_inner_eval):
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
            raise ValueError("Data_idx should be smaller than number of dataset")
        if self.__inner_predict_buffer[data_idx] is None:
            if data_idx == 0:
                n_preds = self.train_set.num_data() * self.__num_class
            else:
                n_preds = self.valid_sets[data_idx - 1].num_data() * self.__num_class
            self.__inner_predict_buffer[data_idx] = \
                np.array([0.0 for _ in range_(n_preds)], dtype=np.float64, copy=False)
        # avoid to predict many time in one iteration
        if not self.__is_predicted_cur_iter[data_idx]:
            tmp_out_len = ctypes.c_int64(0)
            data_ptr = self.__inner_predict_buffer[data_idx].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            _safe_call(_LIB.LGBM_BoosterGetPredict(
                self.handle,
                ctypes.c_int(data_idx),
                ctypes.byref(tmp_out_len),
                data_ptr))
            if tmp_out_len.value != len(self.__inner_predict_buffer[data_idx]):
                raise ValueError("Wrong length of predict results for data %d" % (data_idx))
            self.__is_predicted_cur_iter[data_idx] = True
        return self.__inner_predict_buffer[data_idx]

    def __get_eval_info(self):
        """
        Get inner evaluation count and names
        """
        if self.__need_reload_eval_info:
            self.__need_reload_eval_info = False
            out_num_eval = ctypes.c_int(0)
            # Get num of inner evals
            _safe_call(_LIB.LGBM_BoosterGetEvalCounts(
                self.handle,
                ctypes.byref(out_num_eval)))
            self.__num_inner_eval = out_num_eval.value
            if self.__num_inner_eval > 0:
                # Get name of evals
                tmp_out_len = ctypes.c_int(0)
                string_buffers = [ctypes.create_string_buffer(255) for i in range_(self.__num_inner_eval)]
                ptr_string_buffers = (ctypes.c_char_p * self.__num_inner_eval)(*map(ctypes.addressof, string_buffers))
                _safe_call(_LIB.LGBM_BoosterGetEvalNames(
                    self.handle,
                    ctypes.byref(tmp_out_len),
                    ptr_string_buffers))
                if self.__num_inner_eval != tmp_out_len.value:
                    raise ValueError("Length of eval names doesn't equal with num_evals")
                self.__name_inner_eval = \
                    [string_buffers[i].value.decode() for i in range_(self.__num_inner_eval)]
                self.__higher_better_inner_eval = \
                    [name.startswith(('auc', 'ndcg@', 'map@')) for name in self.__name_inner_eval]

    def attr(self, key):
        """Get attribute string from the Booster.

        Parameters
        ----------
        key : string
            The name of the attribute.

        Returns
        -------
        value : string or None
            The attribute value.
            Returns None if attribute do not exist.
        """
        return self.__attr.get(key, None)

    def set_attr(self, **kwargs):
        """Set the attribute of the Booster.

        Parameters
        ----------
        **kwargs
            The attributes to set.
            Setting a value to None deletes an attribute.
        """
        for key, value in kwargs.items():
            if value is not None:
                if not isinstance(value, string_type):
                    raise ValueError("Set attr only accepts strings")
                self.__attr[key] = value
            else:
                self.__attr.pop(key, None)
