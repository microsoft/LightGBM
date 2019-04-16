/*!
 * Copyright (c) 2017 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <LightGBM/lightgbm_R.h>

#include <LightGBM/utils/common.h>
#include <LightGBM/utils/log.h>
#include <LightGBM/utils/openmp_wrapper.h>
#include <LightGBM/utils/text_reader.h>

#include <string>
#include <cstdio>
#include <cstring>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

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
    R_INT_PTR(call_state)[0] = -1;\
    return call_state;\
  }

using namespace LightGBM;

LGBM_SE EncodeChar(LGBM_SE dest, const char* src, LGBM_SE buf_len, LGBM_SE actual_len, size_t str_len) {
  if (str_len > INT32_MAX) {
    Log::Fatal("Don't support large string in R-package");
  }
  R_INT_PTR(actual_len)[0] = static_cast<int>(str_len);
  if (R_AS_INT(buf_len) < static_cast<int>(str_len)) { return dest; }
  auto ptr = R_CHAR_PTR(dest);
  std::memcpy(ptr, src, str_len);
  return dest;
}

LGBM_SE LGBM_GetLastError_R(LGBM_SE buf_len, LGBM_SE actual_len, LGBM_SE err_msg) {
  return EncodeChar(err_msg, LGBM_GetLastError(), buf_len, actual_len, std::strlen(LGBM_GetLastError()) + 1);
}

LGBM_SE LGBM_DatasetCreateFromFile_R(LGBM_SE filename,
  LGBM_SE parameters,
  LGBM_SE reference,
  LGBM_SE out,
  LGBM_SE call_state) {
  R_API_BEGIN();
  DatasetHandle handle = nullptr;
  CHECK_CALL(LGBM_DatasetCreateFromFile(R_CHAR_PTR(filename), R_CHAR_PTR(parameters),
    R_GET_PTR(reference), &handle));
  R_SET_PTR(out, handle);
  R_API_END();
}

LGBM_SE LGBM_DatasetCreateFromCSC_R(LGBM_SE indptr,
  LGBM_SE indices,
  LGBM_SE data,
  LGBM_SE num_indptr,
  LGBM_SE nelem,
  LGBM_SE num_row,
  LGBM_SE parameters,
  LGBM_SE reference,
  LGBM_SE out,
  LGBM_SE call_state) {
  R_API_BEGIN();
  const int* p_indptr = R_INT_PTR(indptr);
  const int* p_indices = R_INT_PTR(indices);
  const double* p_data = R_REAL_PTR(data);

  int64_t nindptr = static_cast<int64_t>(R_AS_INT(num_indptr));
  int64_t ndata = static_cast<int64_t>(R_AS_INT(nelem));
  int64_t nrow = static_cast<int64_t>(R_AS_INT(num_row));
  DatasetHandle handle = nullptr;
  CHECK_CALL(LGBM_DatasetCreateFromCSC(p_indptr, C_API_DTYPE_INT32, p_indices,
    p_data, C_API_DTYPE_FLOAT64, nindptr, ndata,
    nrow, R_CHAR_PTR(parameters), R_GET_PTR(reference), &handle));
  R_SET_PTR(out, handle);
  R_API_END();
}

LGBM_SE LGBM_DatasetCreateFromMat_R(LGBM_SE data,
  LGBM_SE num_row,
  LGBM_SE num_col,
  LGBM_SE parameters,
  LGBM_SE reference,
  LGBM_SE out,
  LGBM_SE call_state) {
  R_API_BEGIN();
  int32_t nrow = static_cast<int32_t>(R_AS_INT(num_row));
  int32_t ncol = static_cast<int32_t>(R_AS_INT(num_col));
  double* p_mat = R_REAL_PTR(data);
  DatasetHandle handle = nullptr;
  CHECK_CALL(LGBM_DatasetCreateFromMat(p_mat, C_API_DTYPE_FLOAT64, nrow, ncol, COL_MAJOR,
    R_CHAR_PTR(parameters), R_GET_PTR(reference), &handle));
  R_SET_PTR(out, handle);
  R_API_END();
}

LGBM_SE LGBM_DatasetGetSubset_R(LGBM_SE handle,
  LGBM_SE used_row_indices,
  LGBM_SE len_used_row_indices,
  LGBM_SE parameters,
  LGBM_SE out,
  LGBM_SE call_state) {
  R_API_BEGIN();
  int len = R_AS_INT(len_used_row_indices);
  std::vector<int> idxvec(len);
  // convert from one-based to  zero-based index
#pragma omp parallel for schedule(static)
  for (int i = 0; i < len; ++i) {
    idxvec[i] = R_INT_PTR(used_row_indices)[i] - 1;
  }
  DatasetHandle res = nullptr;
  CHECK_CALL(LGBM_DatasetGetSubset(R_GET_PTR(handle),
    idxvec.data(), len, R_CHAR_PTR(parameters),
    &res));
  R_SET_PTR(out, res);
  R_API_END();
}

LGBM_SE LGBM_DatasetSetFeatureNames_R(LGBM_SE handle,
  LGBM_SE feature_names,
  LGBM_SE call_state) {
  R_API_BEGIN();
  auto vec_names = Common::Split(R_CHAR_PTR(feature_names), '\t');
  std::vector<const char*> vec_sptr;
  int len = static_cast<int>(vec_names.size());
  for (int i = 0; i < len; ++i) {
    vec_sptr.push_back(vec_names[i].c_str());
  }
  CHECK_CALL(LGBM_DatasetSetFeatureNames(R_GET_PTR(handle),
    vec_sptr.data(), len));
  R_API_END();
}

LGBM_SE LGBM_DatasetGetFeatureNames_R(LGBM_SE handle,
  LGBM_SE buf_len,
  LGBM_SE actual_len,
  LGBM_SE feature_names,
  LGBM_SE call_state) {
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
  EncodeChar(feature_names, merge_str.c_str(), buf_len, actual_len, merge_str.size() + 1);
  R_API_END();
}

LGBM_SE LGBM_DatasetSaveBinary_R(LGBM_SE handle,
  LGBM_SE filename,
  LGBM_SE call_state) {
  R_API_BEGIN();
  CHECK_CALL(LGBM_DatasetSaveBinary(R_GET_PTR(handle),
    R_CHAR_PTR(filename)));
  R_API_END();
}

LGBM_SE LGBM_DatasetFree_R(LGBM_SE handle,
  LGBM_SE call_state) {
  R_API_BEGIN();
  if (R_GET_PTR(handle) != nullptr) {
    CHECK_CALL(LGBM_DatasetFree(R_GET_PTR(handle)));
    R_SET_PTR(handle, nullptr);
  }
  R_API_END();
}

LGBM_SE LGBM_DatasetSetField_R(LGBM_SE handle,
  LGBM_SE field_name,
  LGBM_SE field_data,
  LGBM_SE num_element,
  LGBM_SE call_state) {
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
  } else if (!strcmp("init_score", name)) {
    CHECK_CALL(LGBM_DatasetSetField(R_GET_PTR(handle), name, R_REAL_PTR(field_data), len, C_API_DTYPE_FLOAT64));
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

LGBM_SE LGBM_DatasetGetField_R(LGBM_SE handle,
  LGBM_SE field_name,
  LGBM_SE field_data,
  LGBM_SE call_state) {
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
  } else if (!strcmp("init_score", name)) {
    auto p_data = reinterpret_cast<const double*>(res);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < out_len; ++i) {
      R_REAL_PTR(field_data)[i] = p_data[i];
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

LGBM_SE LGBM_DatasetGetFieldSize_R(LGBM_SE handle,
  LGBM_SE field_name,
  LGBM_SE out,
  LGBM_SE call_state) {
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

LGBM_SE LGBM_DatasetUpdateParam_R(LGBM_SE handle,
  LGBM_SE params,
  LGBM_SE call_state) {
  R_API_BEGIN();
  CHECK_CALL(LGBM_DatasetUpdateParam(R_GET_PTR(handle), R_CHAR_PTR(params)));
  R_API_END();
}

LGBM_SE LGBM_DatasetGetNumData_R(LGBM_SE handle, LGBM_SE out,
  LGBM_SE call_state) {
  int nrow;
  R_API_BEGIN();
  CHECK_CALL(LGBM_DatasetGetNumData(R_GET_PTR(handle), &nrow));
  R_INT_PTR(out)[0] = static_cast<int>(nrow);
  R_API_END();
}

LGBM_SE LGBM_DatasetGetNumFeature_R(LGBM_SE handle,
  LGBM_SE out,
  LGBM_SE call_state) {
  int nfeature;
  R_API_BEGIN();
  CHECK_CALL(LGBM_DatasetGetNumFeature(R_GET_PTR(handle), &nfeature));
  R_INT_PTR(out)[0] = static_cast<int>(nfeature);
  R_API_END();
}

// --- start Booster interfaces

LGBM_SE LGBM_BoosterFree_R(LGBM_SE handle,
  LGBM_SE call_state) {
  R_API_BEGIN();
  if (R_GET_PTR(handle) != nullptr) {
    CHECK_CALL(LGBM_BoosterFree(R_GET_PTR(handle)));
    R_SET_PTR(handle, nullptr);
  }
  R_API_END();
}

LGBM_SE LGBM_BoosterCreate_R(LGBM_SE train_data,
  LGBM_SE parameters,
  LGBM_SE out,
  LGBM_SE call_state) {
  R_API_BEGIN();
  BoosterHandle handle = nullptr;
  CHECK_CALL(LGBM_BoosterCreate(R_GET_PTR(train_data), R_CHAR_PTR(parameters), &handle));
  R_SET_PTR(out, handle);
  R_API_END();
}

LGBM_SE LGBM_BoosterCreateFromModelfile_R(LGBM_SE filename,
  LGBM_SE out,
  LGBM_SE call_state) {
  R_API_BEGIN();
  int out_num_iterations = 0;
  BoosterHandle handle = nullptr;
  CHECK_CALL(LGBM_BoosterCreateFromModelfile(R_CHAR_PTR(filename), &out_num_iterations, &handle));
  R_SET_PTR(out, handle);
  R_API_END();
}

LGBM_SE LGBM_BoosterLoadModelFromString_R(LGBM_SE model_str,
  LGBM_SE out,
  LGBM_SE call_state) {
  R_API_BEGIN();
  int out_num_iterations = 0;
  BoosterHandle handle = nullptr;
  CHECK_CALL(LGBM_BoosterLoadModelFromString(R_CHAR_PTR(model_str), &out_num_iterations, &handle));
  R_SET_PTR(out, handle);
  R_API_END();
}

LGBM_SE LGBM_BoosterMerge_R(LGBM_SE handle,
  LGBM_SE other_handle,
  LGBM_SE call_state) {
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterMerge(R_GET_PTR(handle), R_GET_PTR(other_handle)));
  R_API_END();
}

LGBM_SE LGBM_BoosterAddValidData_R(LGBM_SE handle,
  LGBM_SE valid_data,
  LGBM_SE call_state) {
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterAddValidData(R_GET_PTR(handle), R_GET_PTR(valid_data)));
  R_API_END();
}

LGBM_SE LGBM_BoosterResetTrainingData_R(LGBM_SE handle,
  LGBM_SE train_data,
  LGBM_SE call_state) {
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterResetTrainingData(R_GET_PTR(handle), R_GET_PTR(train_data)));
  R_API_END();
}

LGBM_SE LGBM_BoosterResetParameter_R(LGBM_SE handle,
  LGBM_SE parameters,
  LGBM_SE call_state) {
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterResetParameter(R_GET_PTR(handle), R_CHAR_PTR(parameters)));
  R_API_END();
}

LGBM_SE LGBM_BoosterGetNumClasses_R(LGBM_SE handle,
  LGBM_SE out,
  LGBM_SE call_state) {
  int num_class;
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterGetNumClasses(R_GET_PTR(handle), &num_class));
  R_INT_PTR(out)[0] = static_cast<int>(num_class);
  R_API_END();
}

LGBM_SE LGBM_BoosterUpdateOneIter_R(LGBM_SE handle,
  LGBM_SE call_state) {
  int is_finished = 0;
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterUpdateOneIter(R_GET_PTR(handle), &is_finished));
  R_API_END();
}

LGBM_SE LGBM_BoosterUpdateOneIterCustom_R(LGBM_SE handle,
  LGBM_SE grad,
  LGBM_SE hess,
  LGBM_SE len,
  LGBM_SE call_state) {
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

LGBM_SE LGBM_BoosterRollbackOneIter_R(LGBM_SE handle,
  LGBM_SE call_state) {
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterRollbackOneIter(R_GET_PTR(handle)));
  R_API_END();
}

LGBM_SE LGBM_BoosterGetCurrentIteration_R(LGBM_SE handle,
  LGBM_SE out,
  LGBM_SE call_state) {
  int out_iteration;
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterGetCurrentIteration(R_GET_PTR(handle), &out_iteration));
  R_INT_PTR(out)[0] = static_cast<int>(out_iteration);
  R_API_END();
}

LGBM_SE LGBM_BoosterGetEvalNames_R(LGBM_SE handle,
  LGBM_SE buf_len,
  LGBM_SE actual_len,
  LGBM_SE eval_names,
  LGBM_SE call_state) {
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
  EncodeChar(eval_names, merge_names.c_str(), buf_len, actual_len, merge_names.size() + 1);
  R_API_END();
}

LGBM_SE LGBM_BoosterGetEval_R(LGBM_SE handle,
  LGBM_SE data_idx,
  LGBM_SE out_result,
  LGBM_SE call_state) {
  R_API_BEGIN();
  int len;
  CHECK_CALL(LGBM_BoosterGetEvalCounts(R_GET_PTR(handle), &len));
  double* ptr_ret = R_REAL_PTR(out_result);
  int out_len;
  CHECK_CALL(LGBM_BoosterGetEval(R_GET_PTR(handle), R_AS_INT(data_idx), &out_len, ptr_ret));
  CHECK(out_len == len);
  R_API_END();
}

LGBM_SE LGBM_BoosterGetNumPredict_R(LGBM_SE handle,
  LGBM_SE data_idx,
  LGBM_SE out,
  LGBM_SE call_state) {
  R_API_BEGIN();
  int64_t len;
  CHECK_CALL(LGBM_BoosterGetNumPredict(R_GET_PTR(handle), R_AS_INT(data_idx), &len));
  R_INT64_PTR(out)[0] = len;
  R_API_END();
}

LGBM_SE LGBM_BoosterGetPredict_R(LGBM_SE handle,
  LGBM_SE data_idx,
  LGBM_SE out_result,
  LGBM_SE call_state) {
  R_API_BEGIN();
  double* ptr_ret = R_REAL_PTR(out_result);
  int64_t out_len;
  CHECK_CALL(LGBM_BoosterGetPredict(R_GET_PTR(handle), R_AS_INT(data_idx), &out_len, ptr_ret));
  R_API_END();
}

int GetPredictType(LGBM_SE is_rawscore, LGBM_SE is_leafidx, LGBM_SE is_predcontrib) {
  int pred_type = C_API_PREDICT_NORMAL;
  if (R_AS_INT(is_rawscore)) {
    pred_type = C_API_PREDICT_RAW_SCORE;
  }
  if (R_AS_INT(is_leafidx)) {
    pred_type = C_API_PREDICT_LEAF_INDEX;
  }
  if (R_AS_INT(is_predcontrib)) {
    pred_type = C_API_PREDICT_CONTRIB;
  }
  return pred_type;
}

LGBM_SE LGBM_BoosterPredictForFile_R(LGBM_SE handle,
  LGBM_SE data_filename,
  LGBM_SE data_has_header,
  LGBM_SE is_rawscore,
  LGBM_SE is_leafidx,
  LGBM_SE is_predcontrib,
  LGBM_SE num_iteration,
  LGBM_SE parameter,
  LGBM_SE result_filename,
  LGBM_SE call_state) {
  R_API_BEGIN();
  int pred_type = GetPredictType(is_rawscore, is_leafidx, is_predcontrib);
  CHECK_CALL(LGBM_BoosterPredictForFile(R_GET_PTR(handle), R_CHAR_PTR(data_filename),
    R_AS_INT(data_has_header), pred_type, R_AS_INT(num_iteration), R_CHAR_PTR(parameter),
    R_CHAR_PTR(result_filename)));
  R_API_END();
}

LGBM_SE LGBM_BoosterCalcNumPredict_R(LGBM_SE handle,
  LGBM_SE num_row,
  LGBM_SE is_rawscore,
  LGBM_SE is_leafidx,
  LGBM_SE is_predcontrib,
  LGBM_SE num_iteration,
  LGBM_SE out_len,
  LGBM_SE call_state) {
  R_API_BEGIN();
  int pred_type = GetPredictType(is_rawscore, is_leafidx, is_predcontrib);
  int64_t len = 0;
  CHECK_CALL(LGBM_BoosterCalcNumPredict(R_GET_PTR(handle), R_AS_INT(num_row),
    pred_type, R_AS_INT(num_iteration), &len));
  R_INT_PTR(out_len)[0] = static_cast<int>(len);
  R_API_END();
}

LGBM_SE LGBM_BoosterPredictForCSC_R(LGBM_SE handle,
  LGBM_SE indptr,
  LGBM_SE indices,
  LGBM_SE data,
  LGBM_SE num_indptr,
  LGBM_SE nelem,
  LGBM_SE num_row,
  LGBM_SE is_rawscore,
  LGBM_SE is_leafidx,
  LGBM_SE is_predcontrib,
  LGBM_SE num_iteration,
  LGBM_SE parameter,
  LGBM_SE out_result,
  LGBM_SE call_state) {
  R_API_BEGIN();
  int pred_type = GetPredictType(is_rawscore, is_leafidx, is_predcontrib);

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
    nrow, pred_type, R_AS_INT(num_iteration), R_CHAR_PTR(parameter), &out_len, ptr_ret));
  R_API_END();
}

LGBM_SE LGBM_BoosterPredictForMat_R(LGBM_SE handle,
  LGBM_SE data,
  LGBM_SE num_row,
  LGBM_SE num_col,
  LGBM_SE is_rawscore,
  LGBM_SE is_leafidx,
  LGBM_SE is_predcontrib,
  LGBM_SE num_iteration,
  LGBM_SE parameter,
  LGBM_SE out_result,
  LGBM_SE call_state) {
  R_API_BEGIN();
  int pred_type = GetPredictType(is_rawscore, is_leafidx, is_predcontrib);

  int32_t nrow = R_AS_INT(num_row);
  int32_t ncol = R_AS_INT(num_col);

  double* p_mat = R_REAL_PTR(data);
  double* ptr_ret = R_REAL_PTR(out_result);
  int64_t out_len;
  CHECK_CALL(LGBM_BoosterPredictForMat(R_GET_PTR(handle),
    p_mat, C_API_DTYPE_FLOAT64, nrow, ncol, COL_MAJOR,
    pred_type, R_AS_INT(num_iteration), R_CHAR_PTR(parameter), &out_len, ptr_ret));

  R_API_END();
}

LGBM_SE LGBM_BoosterSaveModel_R(LGBM_SE handle,
  LGBM_SE num_iteration,
  LGBM_SE filename,
  LGBM_SE call_state) {
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterSaveModel(R_GET_PTR(handle), 0, R_AS_INT(num_iteration), R_CHAR_PTR(filename)));
  R_API_END();
}

LGBM_SE LGBM_BoosterSaveModelToString_R(LGBM_SE handle,
  LGBM_SE num_iteration,
  LGBM_SE buffer_len,
  LGBM_SE actual_len,
  LGBM_SE out_str,
  LGBM_SE call_state) {
  R_API_BEGIN();
  int64_t out_len = 0;
  std::vector<char> inner_char_buf(R_AS_INT(buffer_len));
  CHECK_CALL(LGBM_BoosterSaveModelToString(R_GET_PTR(handle), 0, R_AS_INT(num_iteration), R_AS_INT(buffer_len), &out_len, inner_char_buf.data()));
  EncodeChar(out_str, inner_char_buf.data(), buffer_len, actual_len, static_cast<size_t>(out_len));
  R_API_END();
}

LGBM_SE LGBM_BoosterDumpModel_R(LGBM_SE handle,
  LGBM_SE num_iteration,
  LGBM_SE buffer_len,
  LGBM_SE actual_len,
  LGBM_SE out_str,
  LGBM_SE call_state) {
  R_API_BEGIN();
  int64_t out_len = 0;
  std::vector<char> inner_char_buf(R_AS_INT(buffer_len));
  CHECK_CALL(LGBM_BoosterDumpModel(R_GET_PTR(handle), 0, R_AS_INT(num_iteration), R_AS_INT(buffer_len), &out_len, inner_char_buf.data()));
  EncodeChar(out_str, inner_char_buf.data(), buffer_len, actual_len, static_cast<size_t>(out_len));
  R_API_END();
}
