#include <vector>
#include <string>
#include <utility>
#include <cstring>
#include <cstdio>
#include <sstream>
#include <omp.h>
#include <cstdint>
#include <memory>

#include "./lightgbm_R.h"

#define R_API_BEGIN() \
  GetRNGstate(); \
  try {

#define R_API_END() } \
  catch(std::exception& ex) { PutRNGstate(); error(ex.what()); } \
  catch(std::string& ex) { PutRNGstate(); error(ex.c_str()); } \
  catch(...) { PutRNGstate(); error("unknown exception"); } \
  PutRNGstate();

#ifndef CHECK
#define CHECK(condition)                                   \
  if (!(condition)) error("Check failed: " #condition \
     " at %s, line %d .\n", __FILE__,  __LINE__);
#endif

#define CHECK_CALL(x) \
  if ((x) != 0) { \
    error(LGBM_GetLastError()); \
  }

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
  SEXP ret;
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
    idxvec.data(), len, CHAR(asChar(parameters)),
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
  SEXP ret = R_NilValue;
  R_API_BEGIN();
  const char *name = CHAR(asChar(field_name));
  int64_t out_len = 0;
  int out_type = 0;
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
  UNPROTECT(1);
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

void _BoosterFinalizer(SEXP ext) {
  R_API_BEGIN();
  if (R_ExternalPtrAddr(ext) == NULL) return;
  CHECK_CALL(LGBM_BoosterFree(R_ExternalPtrAddr(ext)));
  R_ClearExternalPtr(ext);
  R_API_END();
}

SEXP LGBM_BoosterCreate_R(SEXP train_data,
  SEXP parameters) {
  SEXP ret;
  R_API_BEGIN();
  BoosterHandle handle;
  CHECK_CALL(LGBM_BoosterCreate(R_ExternalPtrAddr(train_data), CHAR(asChar(parameters)), &handle));
  ret = PROTECT(R_MakeExternalPtr(handle, R_NilValue, R_NilValue));
  R_RegisterCFinalizerEx(ret, _BoosterFinalizer, TRUE);
  R_API_END();
  UNPROTECT(1);
  return ret;
}

SEXP LGBM_BoosterCreateFromModelfile_R(SEXP filename) {
  SEXP ret;
  R_API_BEGIN();
  int64_t out_num_iterations = 0;
  BoosterHandle handle;
  CHECK_CALL(LGBM_BoosterCreateFromModelfile(CHAR(asChar(filename)), &out_num_iterations, &handle));
  ret = PROTECT(R_MakeExternalPtr(handle, R_NilValue, R_NilValue));
  R_RegisterCFinalizerEx(ret, _BoosterFinalizer, TRUE);
  R_API_END();
  UNPROTECT(1);
  return ret;
}

SEXP LGBM_BoosterMerge_R(SEXP handle,
  SEXP other_handle) {
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterMerge(R_ExternalPtrAddr(handle), R_ExternalPtrAddr(other_handle)));
  R_API_END();
  return R_NilValue;
}

SEXP LGBM_BoosterAddValidData_R(SEXP handle,
  SEXP valid_data) {
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterAddValidData(R_ExternalPtrAddr(handle), R_ExternalPtrAddr(valid_data)));
  R_API_END();
  return R_NilValue;
}

SEXP LGBM_BoosterResetTrainingData_R(SEXP handle,
  SEXP train_data) {
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterResetTrainingData(R_ExternalPtrAddr(handle), R_ExternalPtrAddr(train_data)));
  R_API_END();
  return R_NilValue;
}

SEXP LGBM_BoosterResetParameter_R(SEXP handle, SEXP parameters) {
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterResetParameter(R_ExternalPtrAddr(handle), CHAR(asChar(parameters))));
  R_API_END();
  return R_NilValue;
}

SEXP LGBM_BoosterGetNumClasses_R(SEXP handle) {
  int64_t num_class;
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterGetNumClasses(R_ExternalPtrAddr(handle), &num_class));
  R_API_END();
  return ScalarInteger(static_cast<int>(num_class));
}

SEXP LGBM_BoosterUpdateOneIter_R(SEXP handle) {
  int is_finished = 0;
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterUpdateOneIter(R_ExternalPtrAddr(handle), &is_finished));
  R_API_END();
  return ScalarLogical(is_finished == 1);
}

SEXP LGBM_BoosterUpdateOneIterCustom_R(SEXP handle,
  SEXP grad,
  SEXP hess) {
  int is_finished = 0;
  R_API_BEGIN();
  CHECK(length(grad) == length(hess));
  int len = length(grad);
  std::vector<float> tgrad(len), thess(len);
#pragma omp parallel for schedule(static)
  for (int j = 0; j < len; ++j) {
    tgrad[j] = REAL(grad)[j];
    thess[j] = REAL(hess)[j];
  }
  CHECK_CALL(LGBM_BoosterUpdateOneIterCustom(R_ExternalPtrAddr(handle), tgrad.data(), thess.data(), &is_finished));
  R_API_END();
  return ScalarLogical(is_finished == 1);
}

SEXP LGBM_BoosterRollbackOneIter_R(SEXP handle) {
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterRollbackOneIter(R_ExternalPtrAddr(handle)));
  R_API_END();
  return R_NilValue;
}

SEXP LGBM_BoosterGetCurrentIteration_R(SEXP handle) {
  int64_t out_iteration;
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterGetCurrentIteration(R_ExternalPtrAddr(handle), &out_iteration));
  R_API_END();
  return ScalarInteger(static_cast<int>(out_iteration));
}

SEXP LGBM_BoosterGetEvalCounts_R(SEXP handle) {
  int64_t out_len;
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterGetEvalCounts(R_ExternalPtrAddr(handle), &out_len));
  R_API_END();
  return ScalarInteger(static_cast<int>(out_len));
}

SEXP LGBM_BoosterGetEvalNames_R(SEXP handle) {
  SEXP ret;
  R_API_BEGIN();
  int64_t len;
  CHECK_CALL(LGBM_BoosterGetEvalCounts(R_ExternalPtrAddr(handle), &len));
  std::vector<std::unique_ptr<char[]>> names(len);
  std::vector<char*> ptr_names(len);
  for (int i = 0; i < len; ++i) {
    names[i].reset(new char[128]);
    ptr_names[i] = names[i].get();
  }
  int64_t out_len;
  CHECK_CALL(LGBM_BoosterGetEvalNames(R_ExternalPtrAddr(handle), &out_len, ptr_names.data()));
  CHECK(out_len == len);
  ret = PROTECT(allocVector(STRSXP, out_len));
  for (int i = 0; i < out_len; ++i) {
    SET_STRING_ELT(ret, i, mkChar(names[i].get()));
  }
  R_API_END();
  UNPROTECT(1);
  return ret;
}

SEXP LGBM_BoosterGetEval_R(SEXP handle,
  SEXP data_idx) {
  SEXP ret;
  R_API_BEGIN();
  int64_t len;
  CHECK_CALL(LGBM_BoosterGetEvalCounts(R_ExternalPtrAddr(handle), &len));
  std::vector<float> out_result(len);
  int64_t out_len;
  CHECK_CALL(LGBM_BoosterGetEval(R_ExternalPtrAddr(handle), asInteger(data_idx), &out_len, out_result.data()));
  CHECK(out_len == len);
  ret = PROTECT(allocVector(REALSXP, out_len));
  for (int64_t i = 0; i < out_len; ++i) {
    REAL(ret)[i] = out_result[i];
  }
  R_API_END();
  UNPROTECT(1);
  return ret;
}

SEXP LGBM_BoosterGetPredict_R(SEXP handle,
  SEXP data_idx) {
  SEXP ret;
  R_API_BEGIN();
  int64_t len;
  CHECK_CALL(LGBM_BoosterGetNumPredict(R_ExternalPtrAddr(handle), asInteger(data_idx), &len));
  std::vector<float> out_result(len);
  int64_t out_len;
  CHECK_CALL(LGBM_BoosterGetPredict(R_ExternalPtrAddr(handle), asInteger(data_idx), &out_len, out_result.data()));
  CHECK(out_len == len);
  ret = PROTECT(allocVector(REALSXP, out_len));
#pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < out_len; ++i) {
    REAL(ret)[i] = out_result[i];
  }
  R_API_END();
  UNPROTECT(1);
  return ret;
}

SEXP LGBM_BoosterPredictForFile_R(SEXP handle,
  SEXP data_filename,
  SEXP data_has_header,
  SEXP is_rawscore,
  SEXP is_leafidx,
  SEXP num_iteration,
  SEXP result_filename) {
  R_API_BEGIN();
  int pred_type = C_API_PREDICT_NORMAL;
  if (asInteger(is_rawscore)) {
    pred_type = C_API_PREDICT_RAW_SCORE;
  }
  if (asInteger(is_leafidx)) {
    pred_type = C_API_PREDICT_LEAF_INDEX;
  }
  CHECK_CALL(LGBM_BoosterPredictForFile(R_ExternalPtrAddr(handle), CHAR(asChar(data_filename)), 
    asInteger(data_has_header), pred_type, asInteger(num_iteration),
    CHAR(asChar(result_filename)) ));
  R_API_END();
  return R_NilValue;
}

SEXP LGBM_BoosterPredictForCSR_R(SEXP handle,
  SEXP indptr,
  SEXP indices,
  SEXP data,
  SEXP num_col,
  SEXP is_rawscore,
  SEXP is_leafidx,
  SEXP num_iteration) {
  SEXP ret;
  R_API_BEGIN();
  int pred_type = C_API_PREDICT_NORMAL;
  if (asInteger(is_rawscore)) {
    pred_type = C_API_PREDICT_RAW_SCORE;
  }
  if (asInteger(is_leafidx)) {
    pred_type = C_API_PREDICT_LEAF_INDEX;
  }

  const int* p_indptr = INTEGER(indptr);
  const int* p_indices = INTEGER(indices);
  const double* p_data = REAL(data);

  int64_t nindptr = static_cast<int64_t>(length(indptr));
  int64_t ndata = static_cast<int64_t>(length(data));
  int64_t ncol = static_cast<int64_t>(INTEGER(num_col)[0]);
  int64_t nrow = nindptr - 1;
  int64_t len = nrow;
  int num_class = asInteger(LGBM_BoosterGetNumClasses_R(handle));
  len *= num_class;
  if (pred_type == C_API_PREDICT_LEAF_INDEX) {
    int num_iter = asInteger(num_iteration);
    if (num_iter <= 0) {
      num_iter = asInteger(LGBM_BoosterGetCurrentIteration_R(handle));
    }
    len *= num_iter;
  }
  std::vector<float> out_result(len);
  int64_t out_len;

  CHECK_CALL(LGBM_BoosterPredictForCSR(R_ExternalPtrAddr(handle),
    p_indptr, C_API_DTYPE_INT32, p_indices,
    p_data, C_API_DTYPE_FLOAT64, nindptr, ndata,
    ncol, pred_type, asInteger(num_iteration), &out_len, out_result.data()));

  ret = PROTECT(allocVector(REALSXP, out_len));

#pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < out_len; ++i) {
    REAL(ret)[i] = out_result[i];
  }
  R_API_END();
  UNPROTECT(1);
  return ret;
}

SEXP LGBM_BoosterPredictForMat_R(SEXP handle,
  SEXP mat,
  SEXP is_rawscore,
  SEXP is_leafidx,
  SEXP num_iteration) {
  SEXP ret;
  R_API_BEGIN();
  int pred_type = C_API_PREDICT_NORMAL;
  if (asInteger(is_rawscore)) {
    pred_type = C_API_PREDICT_RAW_SCORE;
  }
  if (asInteger(is_leafidx)) {
    pred_type = C_API_PREDICT_LEAF_INDEX;
  }

  SEXP dim = getAttrib(mat, R_DimSymbol);
  int32_t nrow = static_cast<int32_t>(INTEGER(dim)[0]);
  int32_t ncol = static_cast<int32_t>(INTEGER(dim)[1]);
  double* p_mat = REAL(mat);

  int num_class = asInteger(LGBM_BoosterGetNumClasses_R(handle));
  int64_t len = nrow;
  len *= num_class;
  if (pred_type == C_API_PREDICT_LEAF_INDEX) {
    int num_iter = asInteger(num_iteration);
    if (num_iter <= 0) {
      num_iter = asInteger(LGBM_BoosterGetCurrentIteration_R(handle));
    }
    len *= num_iter;
  }

  std::vector<float> out_result(len);
  int64_t out_len;

  CHECK_CALL(LGBM_BoosterPredictForMat(R_ExternalPtrAddr(handle),
    p_mat, C_API_DTYPE_FLOAT64, nrow, ncol, 1,
    pred_type, asInteger(num_iteration), &out_len, out_result.data()));

  ret = PROTECT(allocVector(REALSXP, out_len));

#pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < out_len; ++i) {
    REAL(ret)[i] = out_result[i];
  }

  R_API_END();
  UNPROTECT(1);
  return ret;
}

SEXP LGBM_BoosterSaveModel_R(SEXP handle,
  SEXP num_iteration,
  SEXP filename) {
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterSaveModel(R_ExternalPtrAddr(handle), asInteger(num_iteration), CHAR(asChar(filename))));
  R_API_END();
  return R_NilValue;
}

SEXP LGBM_BoosterDumpModel_R(SEXP handle) {
  std::unique_ptr<char[]> buf;
  R_API_BEGIN();
  int buffer_len = 1024 * 1024;
  buf.reset(new char[buffer_len]);
  int64_t out_len = 0;
  CHECK_CALL(LGBM_BoosterDumpModel(R_ExternalPtrAddr(handle), buffer_len, &out_len, buf.get()));
  if (out_len > buffer_len) {
    buffer_len = out_len;
    buf.reset(new char[buffer_len]);
    CHECK_CALL(LGBM_BoosterDumpModel(R_ExternalPtrAddr(handle), buffer_len, &out_len, buf.get()));
  }
  R_API_END();
  return mkChar(buf.get());
}