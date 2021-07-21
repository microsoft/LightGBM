/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_TREELEARNER_CUDA_CUDA_SPLIT_INFO_HPP_
#define LIGHTGBM_TREELEARNER_CUDA_CUDA_SPLIT_INFO_HPP_

#ifdef USE_CUDA

#include <LightGBM/meta.h>
#include "new_cuda_utils.hpp"

namespace LightGBM {

struct CUDASplitInfo {
 public:
  bool is_valid;
  int leaf_index;
  double gain;
  int inner_feature_index;
  uint32_t threshold;
  bool default_left;

  double left_sum_gradients;
  double left_sum_hessians;
  data_size_t left_count;
  double left_gain;
  double left_value;

  double right_sum_gradients;
  double right_sum_hessians;
  data_size_t right_count;
  double right_gain;
  double right_value;
};

}  // namespace LightGBM

#endif  // USE_CUDA

#endif  // LIGHTGBM_TREELEARNER_CUDA_CUDA_SPLIT_INFO_HPP_
