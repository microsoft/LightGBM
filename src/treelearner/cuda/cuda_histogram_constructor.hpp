/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_CUDA_HISTOGRAM_CONSTRUCTOR_HPP_
#define LIGHTGBM_CUDA_HISTOGRAM_CONSTRUCTOR_HPP_

#ifdef USE_CUDA

#include <LightGBM/feature_group.h>
#include <LightGBM/tree.h>

#include "new_cuda_utils.hpp"

#include <vector>

namespace LightGBM {

class CUDAHistogramConstructor {
 public:
  CUDAHistogramConstructor(const std::vector<int>& feature_group_ids,
    const Dataset* train_data, const int max_num_leaves,
    hist_t* cuda_hist);

  void Init();

  void PushOneData(const uint32_t feature_bin_value, const int feature_group_id, const data_size_t data_index);

  void FinishLoad();

  void ConstructHistogramForLeaf(const int* smaller_leaf_index, const int* larger_leaf_index,
  const data_size_t* num_data_in_smaller_leaf, const int* smaller_leaf_data_offset, const data_size_t* data_indices_ptr,
  const score_t* cuda_gradients, const score_t* cuda_hessians);

  void LaunchConstructHistogramKernel(
  const int* smaller_leaf_index, const data_size_t* num_data_in_leaf, const data_size_t* leaf_num_data_offset,
  const data_size_t* data_indices_ptr, const score_t* cuda_gradients, const score_t* cuda_hessians);

  hist_t* cuda_hist() { return cuda_hist_; }

 private:
  // data on CPU, stored in row-wise style
  std::vector<uint8_t> cpu_data_;
  std::vector<uint32_t> feature_group_bin_offsets_;
  uint8_t* cuda_data_;
  uint32_t cuda_feature_group_bin_offsets_;
  const data_size_t num_data_;
  hist_t* cuda_hist_;
  int num_total_bin_;
  int* cuda_num_total_bin_;
  int num_feature_groups_;
  int* cuda_num_feature_groups_;
  const int max_num_leaves_;
};

class CUDABestSplitFinder {
 public:
  CUDABestSplitFinder(const hist_t* cuda_hist, const Dataset* train_data,
    const std::vector<int>& feature_group_ids, const int max_num_leaves);

  void FindBestSplitsForLeaf(const int* leaf_id);

  void FindBestFromAllSplits();

  int* best_leaf() { return cuda_best_leaf_; }

  int* best_split_feature_index() { return cuda_best_split_feature_index_; }

  int* best_split_threshold() { return cuda_best_split_threshold_; }

 private:
  int* cuda_leaf_best_split_feature_index_;
  int* cuda_leaf_best_split_threshold_;
  double* cuda_leaf_best_split_gain_;

  int* cuda_best_leaf_;
  int* cuda_best_split_feature_index_;
  int* cuda_best_split_threshold_;
};

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_CUDA_HISTOGRAM_CONSTRUCTOR_HPP_
