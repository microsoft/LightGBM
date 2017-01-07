#include <vector>
#include <string>
#include <utility>
#include <cstring>
#include <cstdio>
#include <sstream>
#include <omp.h>
#include <cstdint>
#include <memory>

#include <LightGBM/utils/text_reader.h>
#include <LightGBM/utils/common.h>

#include "./lightgbm_R.h"

#define COL_MAJOR (0)

#define R_API_BEGIN() \
  try {

#define R_API_END() } \
  catch(std::exception& ex) { R_INT_PTR(call_state)[0] = -1; LGBM_SetLastError(ex.what()); return call_state;} \
  catch(std::string& ex) { R_INT_PTR(call_state)[0] = -1; LGBM_SetLastError(ex.c_str()); return call_state; } \
  catch(...) { R_INT_PTR(call_state)[0] = -1; LGBM_SetLastError("unknown exception"); return call_state;} \
  return call_state;

#define CHECK_CALL(x) \
  if ((x) != 0) { \
    R_INT_PTR(call_state)[0] = -1; \
    return call_state; \
  }

using namespace LightGBM;

SEXP EncodeChar(SEXP dest, const char* src, SEXP buf_len, SEXP actual_len) {
  int str_len = static_cast<int>(std::strlen(src));
  R_INT_PTR(actual_len)[0] = str_len;
  if (R_AS_INT(buf_len) < str_len) { return dest; }
  auto ptr = R_CHAR_PTR(dest);
  int i = 0;
  while (src[i] != '\0') {
    ptr[i] = src[i];
    ++i;
  }
  return dest;
}

SEXP LGBM_GetLastError_R(SEXP buf_len, SEXP actual_len, SEXP err_msg) {
  return EncodeChar(err_msg, LGBM_GetLastError(), buf_len, actual_len);
}

SEXP LGBM_DatasetCreateFromFile_R(SEXP filename,
  SEXP parameters,
  SEXP reference,
  SEXP out,
  SEXP call_state) {

  R_API_BEGIN();
  DatasetHandle handle;
  CHECK_CALL(LGBM_DatasetCreateFromFile(R_CHAR_PTR(filename), R_CHAR_PTR(parameters),
    R_GET_PTR(reference), &handle));
  R_SET_PTR(out, handle);
  R_API_END();
}

SEXP LGBM_DatasetCreateFromCSC_R(SEXP indptr,
  SEXP indices,
  SEXP data,
  SEXP num_indptr,
  SEXP nelem,
  SEXP num_row,
  SEXP parameters,
  SEXP reference,
  SEXP out,
  SEXP call_state) {
  R_API_BEGIN();
  const int* p_indptr = R_INT_PTR(indptr);
  const int* p_indices = R_INT_PTR(indices);
  const double* p_data = R_REAL_PTR(data);

  int64_t nindptr = static_cast<int64_t>(R_AS_INT(num_indptr));
  int64_t ndata = static_cast<int64_t>(R_AS_INT(nelem));
  int64_t nrow = static_cast<int64_t>(R_AS_INT(num_row));
  DatasetHandle handle;
  CHECK_CALL(LGBM_DatasetCreateFromCSC(p_indptr, C_API_DTYPE_INT32, p_indices,
    p_data, C_API_DTYPE_FLOAT64, nindptr, ndata,
    nrow, R_CHAR_PTR(parameters), R_GET_PTR(reference), &handle));
  R_SET_PTR(out, handle);
  R_API_END();
}

SEXP LGBM_DatasetCreateFromMat_R(SEXP data,
  SEXP num_row,
  SEXP num_col,
  SEXP parameters,
  SEXP reference,
  SEXP out,
  SEXP call_state) {

  R_API_BEGIN();
  int32_t nrow = static_cast<int32_t>(R_AS_INT(num_row));
  int32_t ncol = static_cast<int32_t>(R_AS_INT(num_col));
  double* p_mat = R_REAL_PTR(data);
  DatasetHandle handle;
  CHECK_CALL(LGBM_DatasetCreateFromMat(p_mat, C_API_DTYPE_FLOAT64, nrow, ncol, COL_MAJOR,
    R_CHAR_PTR(parameters), R_GET_PTR(reference), &handle));
  R_SET_PTR(out, handle);
  R_API_END();
}

SEXP LGBM_DatasetGetSubset_R(SEXP handle,
  SEXP used_row_indices,
  SEXP len_used_row_indices,
  SEXP parameters,
  SEXP out,
  SEXP call_state) {

  R_API_BEGIN();
  int len = R_AS_INT(len_used_row_indices);
  std::vector<int> idxvec(len);
  // convert from one-based to  zero-based index
#pragma omp parallel for schedule(static)
  for (int i = 0; i < len; ++i) {
    idxvec[i] = R_INT_PTR(used_row_indices)[i] - 1;
  }
  DatasetHandle res;
  CHECK_CALL(LGBM_DatasetGetSubset(R_GET_PTR(handle),
    idxvec.data(), len, R_CHAR_PTR(parameters),
    &res));
  R_SET_PTR(out, res);
  R_API_END();
}

SEXP LGBM_DatasetSetFeatureNames_R(SEXP handle,
  SEXP feature_names,
  SEXP call_state) {
  R_API_BEGIN();
  auto vec_names = Common::Split(R_CHAR_PTR(feature_names), "\t");
  std::vector<const char*> vec_sptr;
  int len = static_cast<int>(vec_names.size());
  for (int i = 0; i < len; ++i) {
    vec_sptr.push_back(vec_names[i].c_str());
  }
  CHECK_CALL(LGBM_DatasetSetFeatureNames(R_GET_PTR(handle),
    vec_sptr.data(), len));
  R_API_END();
}

SEXP LGBM_DatasetGetFeatureNames_R(SEXP handle,
  SEXP buf_len,
  SEXP actual_len,
  SEXP feature_names,
  SEXP call_state) {

  R_API_BEGIN();
  int len = 0;
  CHECK_CALL(LGBM_DatasetGetNumFeature(R_GET_PTR(handle), &len));
  std::vector<std::vector<char>> names(len);
  std::vector<char*> ptr_names(len);
  for (int i = 0; i < len; ++i) {
    names[i].resize(256);
    ptr_names[i] = names[i].data();
  }
  int out_len;
  CHECK_CALL(LGBM_DatasetGetFeatureNames(R_GET_PTR(handle),
    ptr_names.data(), &out_len));
  CHECK(len == out_len);
  auto merge_str = Common::Join<char*>(ptr_names, "\t");
  EncodeChar(feature_names, merge_str.c_str(), buf_len, actual_len);
  R_API_END();
}

SEXP LGBM_DatasetSaveBinary_R(SEXP handle,
  SEXP filename,
  SEXP call_state) {
  R_API_BEGIN();
  CHECK_CALL(LGBM_DatasetSaveBinary(R_GET_PTR(handle),
    R_CHAR_PTR(filename)));
  R_API_END();
}

SEXP LGBM_DatasetFree_R(SEXP handle,
  SEXP call_state) {
  R_API_BEGIN();
  if (R_GET_PTR(handle) != nullptr) {
    CHECK_CALL(LGBM_DatasetFree(R_GET_PTR(handle)));
    R_SET_PTR(handle, nullptr);
  }
  R_API_END();
}

SEXP LGBM_DatasetSetField_R(SEXP handle,
  SEXP field_name,
  SEXP field_data,
  SEXP num_element,
  SEXP call_state) {
  R_API_BEGIN();
  int len = static_cast<int>(R_AS_INT(num_element));
  const char* name = R_CHAR_PTR(field_name);
  if (!strcmp("group", name) || !strcmp("query", name)) {
    std::vector<int32_t> vec(len);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < len; ++i) {
      vec[i] = static_cast<int32_t>(R_INT_PTR(field_data)[i]);
    }
    CHECK_CALL(LGBM_DatasetSetField(R_GET_PTR(handle), name, vec.data(), len, C_API_DTYPE_INT32));
  } else {
    std::vector<float> vec(len);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < len; ++i) {
      vec[i] = static_cast<float>(R_REAL_PTR(field_data)[i]);
    }
    CHECK_CALL(LGBM_DatasetSetField(R_GET_PTR(handle), name, vec.data(), len, C_API_DTYPE_FLOAT32));
  }
  R_API_END();
}

SEXP LGBM_DatasetGetField_R(SEXP handle,
  SEXP field_name,
  SEXP field_data,
  SEXP call_state) {

  R_API_BEGIN();
  const char* name = R_CHAR_PTR(field_name);
  int out_len = 0;
  int out_type = 0;
  const void* res;
  CHECK_CALL(LGBM_DatasetGetField(R_GET_PTR(handle), name, &out_len, &res, &out_type));

  if (!strcmp("group", name) || !strcmp("query", name)) {
    auto p_data = reinterpret_cast<const int32_t*>(res);
    // convert from boundaries to size
#pragma omp parallel for schedule(static)
    for (int i = 0; i < out_len - 1; ++i) {
      R_INT_PTR(field_data)[i] = p_data[i + 1] - p_data[i];
    }
  } else {
    auto p_data = reinterpret_cast<const float*>(res);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < out_len; ++i) {
      R_REAL_PTR(field_data)[i] = p_data[i];
    }
  }
  R_API_END();
}

SEXP LGBM_DatasetGetFieldSize_R(SEXP handle,
  SEXP field_name,
  SEXP out,
  SEXP call_state) {

  R_API_BEGIN();
  const char* name = R_CHAR_PTR(field_name);
  int out_len = 0;
  int out_type = 0;
  const void* res;
  CHECK_CALL(LGBM_DatasetGetField(R_GET_PTR(handle), name, &out_len, &res, &out_type));
  if (!strcmp("group", name) || !strcmp("query", name)) {
    out_len -= 1;
  }
  R_INT_PTR(out)[0] = static_cast<int>(out_len);
  R_API_END();
}

SEXP LGBM_DatasetGetNumData_R(SEXP handle, SEXP out,
  SEXP call_state) {
  int nrow;
  R_API_BEGIN();
  CHECK_CALL(LGBM_DatasetGetNumData(R_GET_PTR(handle), &nrow));
  R_INT_PTR(out)[0] = static_cast<int>(nrow);
  R_API_END();
}

SEXP LGBM_DatasetGetNumFeature_R(SEXP handle,
  SEXP out,
  SEXP call_state) {
  int nfeature;
  R_API_BEGIN();
  CHECK_CALL(LGBM_DatasetGetNumFeature(R_GET_PTR(handle), &nfeature));
  R_INT_PTR(out)[0] = static_cast<int>(nfeature);
  R_API_END();
}

// --- start Booster interfaces

SEXP LGBM_BoosterFree_R(SEXP handle,
  SEXP call_state) {
  R_API_BEGIN();
  if (R_GET_PTR(handle) != nullptr) {
    CHECK_CALL(LGBM_BoosterFree(R_GET_PTR(handle)));
    R_SET_PTR(handle, nullptr);
  }
  R_API_END();
}

SEXP LGBM_BoosterCreate_R(SEXP train_data,
  SEXP parameters,
  SEXP out,
  SEXP call_state) {
  R_API_BEGIN();
  BoosterHandle handle;
  CHECK_CALL(LGBM_BoosterCreate(R_GET_PTR(train_data), R_CHAR_PTR(parameters), &handle));
  R_SET_PTR(out, handle);
  R_API_END();
}

SEXP LGBM_BoosterCreateFromModelfile_R(SEXP filename,
  SEXP out,
  SEXP call_state) {

  R_API_BEGIN();
  int out_num_iterations = 0;
  BoosterHandle handle;
  CHECK_CALL(LGBM_BoosterCreateFromModelfile(R_CHAR_PTR(filename), &out_num_iterations, &handle));
  R_SET_PTR(out, handle);
  R_API_END();
}

SEXP LGBM_BoosterMerge_R(SEXP handle,
  SEXP other_handle,
  SEXP call_state) {
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterMerge(R_GET_PTR(handle), R_GET_PTR(other_handle)));
  R_API_END();
}

SEXP LGBM_BoosterAddValidData_R(SEXP handle,
  SEXP valid_data,
  SEXP call_state) {
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterAddValidData(R_GET_PTR(handle), R_GET_PTR(valid_data)));
  R_API_END();
}

SEXP LGBM_BoosterResetTrainingData_R(SEXP handle,
  SEXP train_data,
  SEXP call_state) {
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterResetTrainingData(R_GET_PTR(handle), R_GET_PTR(train_data)));
  R_API_END();
}

SEXP LGBM_BoosterResetParameter_R(SEXP handle,
  SEXP parameters,
  SEXP call_state) {
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterResetParameter(R_GET_PTR(handle), R_CHAR_PTR(parameters)));
  R_API_END();
}

SEXP LGBM_BoosterGetNumClasses_R(SEXP handle,
  SEXP out,
  SEXP call_state) {
  int num_class;
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterGetNumClasses(R_GET_PTR(handle), &num_class));
  R_INT_PTR(out)[0] = static_cast<int>(num_class);
  R_API_END();
}

SEXP LGBM_BoosterUpdateOneIter_R(SEXP handle,
  SEXP call_state) {
  int is_finished = 0;
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterUpdateOneIter(R_GET_PTR(handle), &is_finished));
  R_API_END();
}

SEXP LGBM_BoosterUpdateOneIterCustom_R(SEXP handle,
  SEXP grad,
  SEXP hess,
  SEXP len,
  SEXP call_state) {
  int is_finished = 0;
  R_API_BEGIN();
  int int_len = R_AS_INT(len);
  std::vector<float> tgrad(int_len), thess(int_len);
#pragma omp parallel for schedule(static)
  for (int j = 0; j < int_len; ++j) {
    tgrad[j] = static_cast<float>(R_REAL_PTR(grad)[j]);
    thess[j] = static_cast<float>(R_REAL_PTR(hess)[j]);
  }
  CHECK_CALL(LGBM_BoosterUpdateOneIterCustom(R_GET_PTR(handle), tgrad.data(), thess.data(), &is_finished));
  R_API_END();
}

SEXP LGBM_BoosterRollbackOneIter_R(SEXP handle,
  SEXP call_state) {
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterRollbackOneIter(R_GET_PTR(handle)));
  R_API_END();
}

SEXP LGBM_BoosterGetCurrentIteration_R(SEXP handle,
  SEXP out,
  SEXP call_state) {

  int out_iteration;
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterGetCurrentIteration(R_GET_PTR(handle), &out_iteration));
  R_INT_PTR(out)[0] = static_cast<int>(out_iteration);
  R_API_END();
}

SEXP LGBM_BoosterGetEvalNames_R(SEXP handle,
  SEXP buf_len,
  SEXP actual_len,
  SEXP eval_names,
  SEXP call_state) {

  R_API_BEGIN();
  int len;
  CHECK_CALL(LGBM_BoosterGetEvalCounts(R_GET_PTR(handle), &len));
  std::vector<std::vector<char>> names(len);
  std::vector<char*> ptr_names(len);
  for (int i = 0; i < len; ++i) {
    names[i].resize(128);
    ptr_names[i] = names[i].data();
  }
  int out_len;
  CHECK_CALL(LGBM_BoosterGetEvalNames(R_GET_PTR(handle), &out_len, ptr_names.data()));
  CHECK(out_len == len);
  auto merge_names = Common::Join<char*>(ptr_names, "\t");
  EncodeChar(eval_names, merge_names.c_str(), buf_len, actual_len);
  R_API_END();
}

SEXP LGBM_BoosterGetEval_R(SEXP handle,
  SEXP data_idx,
  SEXP out_result,
  SEXP call_state) {
  R_API_BEGIN();
  int len;
  CHECK_CALL(LGBM_BoosterGetEvalCounts(R_GET_PTR(handle), &len));
  double* ptr_ret = R_REAL_PTR(out_result);
  int out_len;
  CHECK_CALL(LGBM_BoosterGetEval(R_GET_PTR(handle), R_AS_INT(data_idx), &out_len, ptr_ret));
  CHECK(out_len == len);
  R_API_END();
}

SEXP LGBM_BoosterGetNumPredict_R(SEXP handle,
  SEXP data_idx,
  SEXP out,
  SEXP call_state) {
  R_API_BEGIN();
  int64_t len;
  CHECK_CALL(LGBM_BoosterGetNumPredict(R_GET_PTR(handle), R_AS_INT(data_idx), &len));
  R_INT_PTR(out)[0] = static_cast<int>(len);
  R_API_END();
}

SEXP LGBM_BoosterGetPredict_R(SEXP handle,
  SEXP data_idx,
  SEXP out_result,
  SEXP call_state) {
  R_API_BEGIN();
  double* ptr_ret = R_REAL_PTR(out_result);
  int64_t out_len;
  CHECK_CALL(LGBM_BoosterGetPredict(R_GET_PTR(handle), R_AS_INT(data_idx), &out_len, ptr_ret));
  R_API_END();
}

int GetPredictType(SEXP is_rawscore, SEXP is_leafidx) {
  int pred_type = C_API_PREDICT_NORMAL;
  if (R_AS_INT(is_rawscore)) {
    pred_type = C_API_PREDICT_RAW_SCORE;
  }
  if (R_AS_INT(is_leafidx)) {
    pred_type = C_API_PREDICT_LEAF_INDEX;
  }
  return pred_type;
}

SEXP LGBM_BoosterPredictForFile_R(SEXP handle,
  SEXP data_filename,
  SEXP data_has_header,
  SEXP is_rawscore,
  SEXP is_leafidx,
  SEXP num_iteration,
  SEXP result_filename,
  SEXP call_state) {
  R_API_BEGIN();
  int pred_type = GetPredictType(is_rawscore, is_leafidx);
  CHECK_CALL(LGBM_BoosterPredictForFile(R_GET_PTR(handle), R_CHAR_PTR(data_filename),
    R_AS_INT(data_has_header), pred_type, R_AS_INT(num_iteration),
    R_CHAR_PTR(result_filename)));
  R_API_END();
}

SEXP LGBM_BoosterCalcNumPredict_R(SEXP handle,
  SEXP num_row,
  SEXP is_rawscore,
  SEXP is_leafidx,
  SEXP num_iteration,
  SEXP out_len,
  SEXP call_state) {
  R_API_BEGIN();
  int pred_type = GetPredictType(is_rawscore, is_leafidx);
  int64_t len = 0;
  CHECK_CALL(LGBM_BoosterCalcNumPredict(R_GET_PTR(handle), R_AS_INT(num_row),
    pred_type, R_AS_INT(num_iteration), &len));
  R_INT_PTR(out_len)[0] = static_cast<int>(len);
  R_API_END();
}

SEXP LGBM_BoosterPredictForCSC_R(SEXP handle,
  SEXP indptr,
  SEXP indices,
  SEXP data,
  SEXP num_indptr,
  SEXP nelem,
  SEXP num_row,
  SEXP is_rawscore,
  SEXP is_leafidx,
  SEXP num_iteration,
  SEXP out_result,
  SEXP call_state) {

  R_API_BEGIN();
  int pred_type = GetPredictType(is_rawscore, is_leafidx);

  const int* p_indptr = R_INT_PTR(indptr);
  const int* p_indices = R_INT_PTR(indices);
  const double* p_data = R_REAL_PTR(data);

  int64_t nindptr = R_AS_INT(num_indptr);
  int64_t ndata = R_AS_INT(nelem);
  int64_t nrow = R_AS_INT(num_row);
  double* ptr_ret = R_REAL_PTR(out_result);
  int64_t out_len;
  CHECK_CALL(LGBM_BoosterPredictForCSC(R_GET_PTR(handle),
    p_indptr, C_API_DTYPE_INT32, p_indices,
    p_data, C_API_DTYPE_FLOAT64, nindptr, ndata,
    nrow, pred_type, R_AS_INT(num_iteration), &out_len, ptr_ret));
  R_API_END();
}

SEXP LGBM_BoosterPredictForMat_R(SEXP handle,
  SEXP data,
  SEXP num_row,
  SEXP num_col,
  SEXP is_rawscore,
  SEXP is_leafidx,
  SEXP num_iteration,
  SEXP out_result,
  SEXP call_state) {

  R_API_BEGIN();
  int pred_type = GetPredictType(is_rawscore, is_leafidx);

  int32_t nrow = R_AS_INT(num_row);
  int32_t ncol = R_AS_INT(num_col);

  double* p_mat = R_REAL_PTR(data);
  double* ptr_ret = R_REAL_PTR(out_result);
  int64_t out_len;
  CHECK_CALL(LGBM_BoosterPredictForMat(R_GET_PTR(handle),
    p_mat, C_API_DTYPE_FLOAT64, nrow, ncol, COL_MAJOR,
    pred_type, R_AS_INT(num_iteration), &out_len, ptr_ret));

  R_API_END();
}

SEXP LGBM_BoosterSaveModel_R(SEXP handle,
  SEXP num_iteration,
  SEXP filename,
  SEXP call_state) {
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterSaveModel(R_GET_PTR(handle), R_AS_INT(num_iteration), R_CHAR_PTR(filename)));
  R_API_END();
}

SEXP LGBM_BoosterDumpModel_R(SEXP handle,
  SEXP num_iteration,
  SEXP buffer_len,
  SEXP actual_len,
  SEXP out_str,
  SEXP call_state) {
  R_API_BEGIN();
  int out_len = 0;
  std::vector<char> inner_char_buf(R_AS_INT(buffer_len));
  CHECK_CALL(LGBM_BoosterDumpModel(R_GET_PTR(handle), R_AS_INT(num_iteration), R_AS_INT(buffer_len), &out_len, inner_char_buf.data()));
  EncodeChar(out_str, inner_char_buf.data(), buffer_len, actual_len);
  if (out_len < R_AS_INT(buffer_len)) {
    EncodeChar(out_str, inner_char_buf.data(), buffer_len, actual_len);
  } else {
    R_INT_PTR(actual_len)[0] = static_cast<int>(out_len);
  }
  R_API_END();
}
