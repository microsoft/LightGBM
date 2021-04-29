/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_CUDA_BEST_SPLIT_FINDER_HPP_
#define LIGHTGBM_CUDA_BEST_SPLIT_FINDER_HPP_

#include "new_cuda_utils.hpp"

#include <LightGBM/bin.h>
#include <LightGBM/dataset.h>

#include <vector>

#ifdef USE_CUDA

namespace LightGBM {

class CUDABestSplitFinder {
 public:
  CUDABestSplitFinder(const hist_t* cuda_hist, const Dataset* train_data,
    const std::vector<uint32_t>& feature_hist_offsets, const int max_num_leaves);

  void Init();

  void FindBestSplitsForLeaf(const int* smaller_leaf_id, const int* larger_leaf_id, const double* parent_gain);

  void FindBestFromAllSplits();

  int* best_leaf() { return cuda_best_leaf_; }

  int* best_split_feature_index() { return cuda_best_split_feature_index_; }

  int* best_split_threshold() { return cuda_best_split_threshold_; }

 private:
  void LaunchFindBestSplitsForLeafKernel(const int* smaller_leaf_id, const int* larger_leaf_id, const double* parent_gain);

  int* cuda_leaf_best_split_feature_index_;
  int* cuda_leaf_best_split_threshold_;
  double* cuda_leaf_best_split_gain_;

  int* cuda_best_leaf_;
  int* cuda_best_split_feature_index_;
  int* cuda_best_split_threshold_;

  double* cuda_leaf_best_split_gain_;
  int* cuda_leaf_best_split_feature_;
  int* cuda_leaf_best_split_threshold_;

  int* cuda_best_split_feature_;
  uint8_t* cuda_best_split_default_left_;
  double* cuda_best_split_gain_;
  double* cuda_best_split_left_sum_gradient_;
  double* cuda_best_split_left_sum_hessian_;
  data_size_t* cuda_best_split_left_count_;
  double* cuda_best_split_right_sum_gradient_;
  double* cuda_best_split_right_sum_hessian_;
  data_size_t* cuda_best_split_right_count_;

  const hist_t* cuda_hist_;
  hist_t* prefix_sum_hist_left_;
  hist_t* prefix_sum_hist_right_;
  const int num_features_;
  const int max_num_leaves_;
  const int num_total_bin_;

  int* cuda_num_total_bin_;

  std::vector<uint32_t> feature_hist_offsets_;
  std::vector<uint8_t> feature_mfb_offsets_;
  std::vector<uint32_t> feature_default_bins_;

  // None --> 0, Zero --> 1, NaN --> 2
  std::vector<uint8_t> feature_missing_type_;
  const double lambda_l1_;
  const data_size_t min_data_in_leaf_;
  const double min_sum_hessian_in_leaf_;
  const double min_gain_to_split_;

  uint32_t* cuda_feature_hist_offsets_;
  uint8_t* cuda_feature_mfb_offsets_;
  uint32_t* cuda_feature_default_bins_;
  uint8_t* cuda_feature_missing_type_;
  double* cuda_lambda_l1_;
  data_size_t* cuda_min_data_in_leaf_;
  double* cuda_min_sum_hessian_in_leaf_;
  double* cuda_min_gain_to_split_;
};


}

#endif  // USE_CUDA
#endif  // LIGHTGBM_CUDA_HISTOGRAM_CONSTRUCTOR_HPP_
