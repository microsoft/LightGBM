/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_tree_predictor.hpp"

namespace LightGBM {

CUDATreePredictor::CUDATreePredictor(const Config* config,
  const int* tree_split_leaf_index,
  const int* tree_inner_feature_index,
  const uint32_t* tree_threshold,
  const double* tree_threshold_real,
  const double* tree_left_output,
  const double* tree_right_output,
  const data_size_t* tree_left_count,
  const data_size_t* tree_right_count,
  const double* tree_left_sum_hessian,
  const double* tree_right_sum_hessian,
  const double* tree_gain,
  const uint8_t* tree_default_left,
  const double* leaf_output):
tree_split_leaf_index_(tree_split_leaf_index),
tree_inner_feature_index_(tree_inner_feature_index),
tree_threshold_(tree_threshold),
  {

}

}  // namespace LightGBM

#endif  // USE_CUDA
