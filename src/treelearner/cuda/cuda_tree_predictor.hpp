/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_CUDA_TREE_PREDICTOR_HPP_
#define LIGHTGBM_CUDA_TREE_PREDICTOR_HPP_

#ifdef USE_CUDA

#include <LightGBM/meta.h>
#include <LightGBM/config.h>
#include "new_cuda_utils.hpp"

#include <vector>

namespace LightGBM {

class CUDATreePredictor {
 public:
  CUDATreePredictor(const Config* config,
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
    const double* leaf_output);

  void Predict(const double* data, double* out_score) const;

 private:
  void BuildTree();

  void LaunchPredictKernel(const double* data, double* out_score) const;

  // CUDA memory, held by other objects 
  const int* tree_split_leaf_index_;
  const int* tree_inner_feature_index_;
  const uint32_t* tree_threshold_;
  const double* tree_threshold_real_;
  const double* tree_left_output_;
  const double* tree_right_output_;
  const data_size_t* tree_left_count_;
  const data_size_t* tree_right_count_;
  const double* tree_left_sum_hessian_;
  const double* tree_right_sum_hessian_;
  const double* tree_gain_;
  const uint8_t* tree_default_left_;
  const double* leaf_output_;
};

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_CUDA_TREE_PREDICTOR_HPP_
