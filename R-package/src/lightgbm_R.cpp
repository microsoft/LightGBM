#include <vector>
#include <string>
#include <utility>
#include <cstring>
#include <cstdio>
#include <sstream>
#include <omp.h>
#include <cstdint>

#include "./lightgbm_R.h"

#define R_API_BEGIN() \
  GetRNGstate(); \
  try {

#define R_API_END() } \
  catch(std::exception& ex) { PutRNGstate(); error(ex.what()); } \
  catch(std::string& ex) { PutRNGstate(); error(ex.c_str()); } \
  catch(...) { PutRNGstate(); error("unknown exception"); } \
  PutRNGstate();

#define CHECK_CALL(x) \
  if ((x) != 0) { \
    error(LGBM_GetLastError()); \
  }

using namespace LightGBM;

SEXP LGBMCheckNullPtr_R(SEXP handle) {
  return ScalarLogical(R_ExternalPtrAddr(handle) == NULL);
}

void _DatasetFinalizer(SEXP ext) {
  R_API_BEGIN();
  if (R_ExternalPtrAddr(ext) == NULL) return;
  CHECK_CALL(LGBM_DatasetFree(R_ExternalPtrAddr(ext)));
  R_ClearExternalPtr(ext);
  R_API_END();
}

SEXP LGBM_DatasetCreateFromFile_R(SEXP filename, SEXP parameters, SEXP reference) {
  SEXP ret;
  R_API_BEGIN();
  DatasetHandle handle;
  CHECK_CALL(LGBM_DatasetCreateFromFile(CHAR(asChar(filename)), CHAR(asChar(parameters)), 
    R_ExternalPtrAddr(reference), &handle));
  ret = PROTECT(R_MakeExternalPtr(handle, R_NilValue, R_NilValue));
  R_RegisterCFinalizerEx(ret, _DatasetFinalizer, TRUE);
  R_API_END();
  UNPROTECT(1);
  return ret;
}

SEXP LGBM_DatasetCreateFromCSR_R(SEXP indptr,
  SEXP indices,
  SEXP data,
  SEXP num_col,
  SEXP parameters,
  SEXP reference) {
  SEXP ret;
  R_API_BEGIN();
  const int* p_indptr = INTEGER(indptr);
  const int* p_indices = INTEGER(indices);
  const double* p_data = REAL(data);

  int64_t nindptr = static_cast<int64_t>(length(indptr));
  int64_t ndata = static_cast<int64_t>(length(data));
  int64_t ncol = static_cast<int64_t>(INTEGER(num_col)[0]);

  DatasetHandle handle;
  CHECK_CALL(LGBM_DatasetCreateFromCSR(p_indptr, C_API_DTYPE_INT32, p_indices,
    p_data, C_API_DTYPE_FLOAT64, nindptr, ndata,
    ncol, CHAR(asChar(parameters)), R_ExternalPtrAddr(reference), &handle));
  ret = PROTECT(R_MakeExternalPtr(handle, R_NilValue, R_NilValue));
  R_RegisterCFinalizerEx(ret, _DatasetFinalizer, TRUE);
  R_API_END();
  UNPROTECT(1);
  return ret;
}

SEXP LGBM_DatasetCreateFromMat_R(SEXP mat,
  SEXP parameters,
  SEXP reference) {
  SEXP ret;
  R_API_BEGIN();
  SEXP dim = getAttrib(mat, R_DimSymbol);
  int32_t nrow = static_cast<int32_t>(INTEGER(dim)[0]);
  int32_t ncol = static_cast<int32_t>(INTEGER(dim)[1]);
  double* p_mat = REAL(mat);

  DatasetHandle handle;
  CHECK_CALL(LGBM_DatasetCreateFromMat(p_mat, C_API_DTYPE_FLOAT64, nrow, ncol, 1,
    CHAR(asChar(parameters)), R_ExternalPtrAddr(reference), &handle));
  ret = PROTECT(R_MakeExternalPtr(handle, R_NilValue, R_NilValue));
  R_RegisterCFinalizerEx(ret, _DatasetFinalizer, TRUE);
  R_API_END();
  UNPROTECT(1);
  return ret;
}

SEXP LGBM_DatasetGetSubset_R(SEXP handle,
  SEXP used_row_indices,
  SEXP parameters) {
  R_API_BEGIN();
  int len = length(used_row_indices);
  std::vector<int> idxvec(len);
  // convert from one-based to  zero-based index
#pragma omp parallel for schedule(static)
  for (int i = 0; i < len; ++i) {
    idxvec[i] = INTEGER(used_row_indices)[i] - 1;
  }
  DatasetHandle res;
  CHECK_CALL(LGBM_DatasetGetSubset(R_ExternalPtrAddr(handle),
    idxvec.data(), len, CHAR(asChar(parameters))
    &res));
  ret = PROTECT(R_MakeExternalPtr(res, R_NilValue, R_NilValue));
  R_RegisterCFinalizerEx(ret, _DatasetFinalizer, TRUE);
  R_API_END();
  UNPROTECT(1);
  return ret;
}

SEXP LGBM_DatasetSetFeatureNames_R(SEXP handle,
  SEXP feature_names) {
  R_API_BEGIN();
  std::vector<std::string> vec_names;
  std::vector<const char*> vec_sptr;
  int64_t len = static_cast<int64_t>(length(feature_names));
  for (int i = 0; i < len; ++i) {
    vec_names.push_back(std::string(CHAR(asChar(VECTOR_ELT(feature_names, i)))));
  }
  for (int i = 0; i < len; ++i) {
    vec_sptr.push_back(vec_names[i].c_str());
  }
  CHECK_CALL(LGBM_DatasetSetFeatureNames(R_ExternalPtrAddr(handle),
    vec_sptr.data(), len));
  R_API_END();
  return R_NilValue;
}

SEXP LGBM_DatasetSaveBinary_R(SEXP handle,
  SEXP filename) {
  R_API_BEGIN();
  CHECK_CALL(LGBM_DatasetSaveBinary(R_ExternalPtrAddr(handle),
    CHAR(asChar(filename))));
  R_API_END();
  return R_NilValue;
}

SEXP LGBM_DatasetSetField_R(SEXP handle,
  SEXP field_name,
  SEXP field_data) {
  R_API_BEGIN();
  int64_t len = static_cast<int64_t>(length(field_data));
  const char *name = CHAR(asChar(field_name));
  if (!strcmp("group", name) || !strcmp("query", name)) {
    std::vector<int32_t> vec(len);
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < len; ++i) {
      vec[i] = static_cast<int32_t>(INTEGER(field_data)[i]);
    }
    CHECK_CALL(LGBM_DatasetSetField(R_ExternalPtrAddr(handle), name,  vec.data(), len, C_API_DTYPE_INT32));
  } else {
    std::vector<float> vec(len);
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < len; ++i) {
      vec[i] = static_cast<float>(REAL(field_data)[i]);
    }
    CHECK_CALL(LGBM_DatasetSetField(R_ExternalPtrAddr(handle), name, vec.data(), len, C_API_DTYPE_FLOAT32));
  }
  R_API_END();
  return R_NilValue;
}

SEXP LGBM_DatasetGetField_R(SEXP handle,
  SEXP field_name) {
  R_API_BEGIN();
  const char *name = CHAR(asChar(field_name));
  int64_t out_len = 0;
  int out_type = 0;
  SEXP ret = R_NilValue;
  const void* res;
  CHECK_CALL(LGBM_DatasetGetField(R_ExternalPtrAddr(handle), name, &out_len, &res, &out_type));

  if (out_type == C_API_DTYPE_INT32) {
    ret = PROTECT(allocVector(INTSXP, out_len));
    auto p_data = reinterpret_cast<const int32_t*>(res);
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < out_len; ++i) {
      INTEGER(ret)[i] = p_data[i];
    }
  } else {
    ret = PROTECT(allocVector(REALSXP, out_len));
    auto p_data = reinterpret_cast<const float*>(res);
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < out_len; ++i) {
      REAL(ret)[i] = p_data[i];
    }
  }
  R_API_END();
  return ret;
}

SEXP LGBM_DatasetGetNumData_R(SEXP handle) {
  int64_t nrow;
  R_API_BEGIN();
  CHECK_CALL(LGBM_DatasetGetNumData(R_ExternalPtrAddr(handle), &nrow));
  R_API_END();
  return ScalarInteger(static_cast<int>(nrow));
}

SEXP LGBM_DatasetGetNumFeature_R(SEXP handle) {
  int64_t nfeature;
  R_API_BEGIN();
  CHECK_CALL(LGBM_DatasetGetNumFeature(R_ExternalPtrAddr(handle), &nfeature));
  R_API_END();
  return ScalarInteger(static_cast<int>(nfeature));
}

// --- start Booster interfaces

/*!
* \brief create an new boosting learner
* \param train_data training data set
* \param parameters format: 'key1=value1 key2=value2'
* \return out created Booster
*/
SEXP LGBM_BoosterCreate_R(SEXP train_data,
  SEXP parameters);

/*!
* \brief load an existing boosting from model file
* \param filename filename of model
* \return handle of created Booster
*/
SEXP LGBM_BoosterCreateFromModelfile_R(SEXP filename);

/*!
* \brief Merge model in two booster to first handle
* \param handle handle, will merge other handle to this
* \param other_handle
* \return R_NilValue
*/
SEXP LGBM_BoosterMerge_R(SEXP handle,
  SEXP other_handle);

/*!
* \brief Add new validation to booster
* \param handle handle
* \param valid_data validation data set
* \return R_NilValue
*/
SEXP LGBM_BoosterAddValidData_R(SEXP handle,
  SEXP valid_data);

/*!
* \brief Reset training data for booster
* \param handle handle
* \param train_data training data set
* \return R_NilValue
*/
SEXP LGBM_BoosterResetTrainingData_R(SEXP handle,
  SEXP train_data);

/*!
* \brief Reset config for current booster
* \param handle handle
* \param parameters format: 'key1=value1 key2=value2'
* \return R_NilValue
*/
SEXP LGBM_BoosterResetParameter_R(SEXP handle, SEXP parameters);

/*!
* \brief Get number of class
* \param handle handle
* \return number of classes
*/
SEXP LGBM_BoosterGetNumClasses_R(SEXP handle);

/*!
* \brief update the model in one round
* \param handle handle
* \return bool, true means finished
*/
SEXP LGBM_BoosterUpdateOneIter_R(SEXP handle);

/*!
* \brief update the model, by directly specify gradient and second order gradient,
*       this can be used to support customized loss function
* \param handle handle
* \param grad gradient statistics
* \param hess second order gradient statistics
* \return bool, true means finished
*/
SEXP LGBM_BoosterUpdateOneIterCustom_R(SEXP handle,
  SEXP grad,
  SEXP hess);

/*!
* \brief Rollback one iteration
* \param handle handle
* \return R_NilValue
*/
SEXP LGBM_BoosterRollbackOneIter_R(SEXP handle);

/*!
* \brief Get iteration of current boosting rounds
* \return iteration of boosting rounds
*/
SEXP LGBM_BoosterGetCurrentIteration_R(SEXP handle);

/*!
* \brief Get number of eval
* \return total number of eval results
*/
SEXP LGBM_BoosterGetEvalCounts_R(SEXP handle);

/*!
* \brief Get Name of eval
* \return out_strs names of eval result
*/
SEXP LGBM_BoosterGetEvalNames_R(SEXP handle);

/*!
* \brief get evaluation for training data and validation data
* \param handle handle
* \param data_idx 0:training data, 1: 1st valid data, 2:2nd valid data ...
* \return float arrary contains result
*/
SEXP LGBM_BoosterGetEval_R(SEXP handle,
  SEXP data_idx);

/*!
* \brief Get prediction for training data and validation data
this can be used to support customized eval function
* \param handle handle
* \param data_idx 0:training data, 1: 1st valid data, 2:2nd valid data ...
* \return prediction result
*/
SEXP LGBM_BoosterGetPredict_R(SEXP handle,
  SEXP data_idx);

/*!
* \brief make prediction for file
* \param handle handle
* \param data_filename filename of data file
* \param data_has_header data file has header or not
* \param predict_type
*          C_API_PREDICT_NORMAL: normal prediction, with transform _R(if needed)
*          C_API_PREDICT_RAW_SCORE: raw score
*          C_API_PREDICT_LEAF_INDEX: leaf index
* \param num_iteration number of iteration for prediction, <= 0 means no limit
* \param result_filename filename of result file
* \return R_NilValue
*/
SEXP LGBM_BoosterPredictForFile_R(SEXP handle,
  SEXP data_filename,
  SEXP data_has_header,
  SEXP predict_type,
  SEXP num_iteration,
  SEXP result_filename);

/*!
* \brief make prediction for an new data set
*        Note:  should pre-allocate memory for out_result,
*               for noraml and raw score: its length is equal to num_class * num_data
*               for leaf index, its length is equal to num_class * num_data * num_iteration
* \param handle handle
* \param indptr pointer to row headers
* \param indices findex
* \param data fvalue
* \param num_col number of columns
* \param predict_type
*          C_API_PREDICT_NORMAL: normal prediction, with transform _R(if needed)
*          C_API_PREDICT_RAW_SCORE: raw score
*          C_API_PREDICT_LEAF_INDEX: leaf index
* \param num_iteration number of iteration for prediction, <= 0 means no limit
* \return prediction result
*/
SEXP LGBM_BoosterPredictForCSR_R(SEXP handle,
  SEXP indptr,
  SEXP indices,
  SEXP data,
  SEXP num_col,
  SEXP predict_type,
  SEXP num_iteration);

/*!
* \brief make prediction for an new data set
*        Note:  should pre-allocate memory for out_result,
*               for noraml and raw score: its length is equal to num_class * num_data
*               for leaf index, its length is equal to num_class * num_data * num_iteration
* \param handle handle
* \param data pointer to the data space
* \param predict_type
*          C_API_PREDICT_NORMAL: normal prediction, with transform _R(if needed)
*          C_API_PREDICT_RAW_SCORE: raw score
*          C_API_PREDICT_LEAF_INDEX: leaf index
* \param num_iteration number of iteration for prediction, <= 0 means no limit
* \return prediction result
*/
SEXP LGBM_BoosterPredictForMat_R(SEXP handle,
  SEXP data,
  SEXP predict_type,
  SEXP num_iteration);

/*!
* \brief save model into file
* \param handle handle
* \param num_iteration, <= 0 means save all
* \param filename file name
* \return R_NilValue
*/
SEXP LGBM_BoosterSaveModel_R(SEXP handle,
  SEXP num_iteration,
  SEXP filename);

/*!
* \brief dump model to json
* \param handle handle
* \return json format string of model
*/
SEXP LGBM_BoosterDumpModel_R(SEXP handle);