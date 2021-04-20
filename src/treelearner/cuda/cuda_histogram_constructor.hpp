/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_NEW_CUDA_HISTOGRAM_CONSTRUCTOR_HPP_
#define LIGHTGBM_NEW_CUDA_HISTOGRAM_CONSTRUCTOR_HPP_

#ifdef USE_CUDA

#include <LightGBM/feature_group.h>

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

  void ConstructHistogramForLeaf(const int* smaller_leaf_index, const int* larger_leaf_index);

  hist_t* cuda_hist() { return cuda_hist_; }

 private:
  // data on CPU, stored in row-wise style
  std::vector<uint8_t> cpu_data_;
  std::vector<uint32_t> feature_group_bin_offsets;
  uint8_t* cuda_data_;
  const data_size_t num_data_;
  hist_t* cuda_hist_;
};

class CUDALeafSplitsInit {
 public:
  CUDALeafSplitsInit(const score_t* cuda_gradients, const score_t* cuda_hessians, const data_size_t num_data);

  void Init();

  const double* smaller_leaf_sum_gradients() { return smaller_leaf_sum_gradients_; }

  const double* smaller_leaf_sum_hessians() { return smaller_leaf_sum_hessians_; }

  const double* larger_leaf_sum_gradients() { return larger_leaf_sum_gradients_; }

  const double* larger_leaf_sum_gradients() { return larger_leaf_sum_hessians_; }

  const int* smaller_leaf_index() { return smaller_leaf_index_; }
  
  const int* larger_leaf_index() { return larger_leaf_index_; }

 protected:
  const score_t* cuda_gradients_;
  const score_t* cuda_hessians_;
  double* smaller_leaf_sum_gradients_;
  double* smaller_leaf_sum_hessians_;
  double* larger_leaf_sum_gradients_;
  double* larger_leaf_sum_hessians_;
  int* smaller_leaf_index_;
  int* larger_leaf_index_;

  int num_cuda_blocks_;
  const int num_data_;
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

class CUDADataSplitter {
 public:
  CUDADataSplitter(const data_size_t* data_indices, const data_size_t num_data);

  void Init();

  void Split(const int* leaf_id, const int* best_split_feature, const int* best_split_threshold);

  Tree* GetCPUTree();
};

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_NEW_CUDA_HISTOGRAM_CONSTRUCTOR_HPP_
