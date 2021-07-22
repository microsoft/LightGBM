/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifndef LIGHTGBM_CUDA_BEST_SPLIT_FINDER_HPP_
#define LIGHTGBM_CUDA_BEST_SPLIT_FINDER_HPP_

#ifdef USE_CUDA

#include "new_cuda_utils.hpp"
#include "cuda_leaf_splits.hpp"

#include <LightGBM/bin.h>
#include <LightGBM/cuda/cuda_split_info.hpp>
#include <LightGBM/dataset.h>

#include <vector>

#define MAX_NUM_BIN_IN_FEATURE (256)
#define NUM_THREADS_FIND_BEST_LEAF (256)
#define LOG_NUM_BANKS_DATA_PARTITION_BEST_SPLIT_FINDER (4)
#define NUM_BANKS_DATA_PARTITION_BEST_SPLIT_FINDER (16)
#define NUM_TASKS_PER_SYNC_BLOCK (1024)

namespace LightGBM {

class CUDABestSplitFinder {
 public:
  CUDABestSplitFinder(const hist_t* cuda_hist, const Dataset* train_data,
    const std::vector<uint32_t>& feature_hist_offsets, const int num_leaves,
    const double lambda_l1, const double lambda_l2, const data_size_t min_data_in_leaf,
    const double min_sum_hessian_in_leaf, const double min_gain_to_split,
    const int* cuda_num_features);

  void Init();

  void BeforeTrain();

  void FindBestSplitsForLeaf(const CUDALeafSplitsStruct* smaller_leaf_splits, const CUDALeafSplitsStruct* larger_leaf_splits,
    const int smaller_leaf_index, const int larger_leaf_index,
    const data_size_t num_data_in_smaller_leaf, const data_size_t num_data_in_larger_leaf,
    const double sum_hessians_in_smaller_leaf, const double sum_hessians_in_larger_leaf);

  const CUDASplitInfo* FindBestFromAllSplits(const int cur_num_leaves, const int smaller_leaf_index,
    const int larger_leaf_index, std::vector<int>* leaf_best_split_feature,
    std::vector<uint32_t>* leaf_best_split_threshold, std::vector<uint8_t>* leaf_best_split_default_left, int* best_leaf_index);

  const int* cuda_best_leaf() const { return cuda_best_leaf_; }

  CUDASplitInfo* cuda_leaf_best_split_info() { return cuda_leaf_best_split_info_; }

 private:
  void LaunchFindBestSplitsForLeafKernel(const CUDALeafSplitsStruct* smaller_leaf_splits,
    const CUDALeafSplitsStruct* larger_leaf_splits, const int smaller_leaf_index, const int larger_leaf_index,
    const bool is_smaller_leaf_valid, const bool is_larger_leaf_valid);

  void LaunchSyncBestSplitForLeafKernel(
    const int cpu_smaller_leaf_index,
    const int cpu_larger_leaf_index,
    const bool is_smaller_leaf_valid,
    const bool is_larger_leaf_valid);

  void LaunchFindBestFromAllSplitsKernel(const int cur_num_leaves, const int smaller_leaf_index,
    const int larger_leaf_index, std::vector<int>* leaf_best_split_feature,
    std::vector<uint32_t>* leaf_best_split_threshold, std::vector<uint8_t>* leaf_best_split_default_left, int* best_leaf_index);

  // Host memory
  const int num_features_;
  const int num_leaves_;
  const int num_total_bin_;
  int max_num_bin_in_feature_;
  std::vector<uint32_t> feature_hist_offsets_;
  std::vector<uint8_t> feature_mfb_offsets_;
  std::vector<uint32_t> feature_default_bins_;
  std::vector<uint32_t> feature_num_bins_;
  // None --> 0, Zero --> 1, NaN --> 2
  std::vector<uint8_t> feature_missing_type_;
  const double lambda_l1_;
  const double lambda_l2_;
  const data_size_t min_data_in_leaf_;
  const double min_sum_hessian_in_leaf_;
  const double min_gain_to_split_;
  std::vector<cudaStream_t> cuda_streams_;
  // for best split find tasks
  std::vector<int> cpu_task_feature_index_;
  std::vector<uint8_t> cpu_task_reverse_;
  std::vector<uint8_t> cpu_task_skip_default_bin_;
  std::vector<uint8_t> cpu_task_na_as_missing_;
  std::vector<uint8_t> cpu_task_out_default_left_;
  int num_tasks_;

  // CUDA memory, held by this object
  // for per leaf best split information
  int* cuda_best_leaf_;
  CUDASplitInfo* cuda_leaf_best_split_info_;
  // for best split information when finding best split
  CUDASplitInfo* cuda_best_split_info_;
  int* cuda_num_total_bin_;
  // TODO(shiyu1994): use prefix sum to accelerate best split finding
  hist_t* prefix_sum_hist_left_;
  hist_t* prefix_sum_hist_right_;
  // feature information
  uint32_t* cuda_feature_hist_offsets_;
  uint8_t* cuda_feature_mfb_offsets_;
  uint32_t* cuda_feature_default_bins_;
  uint8_t* cuda_feature_missing_type_;
  uint32_t* cuda_feature_num_bins_;
  // best split information buffer, to be copied to CPU
  int* cuda_best_split_info_buffer_;
  // find best split task information
  int* cuda_task_feature_index_;
  uint8_t* cuda_task_reverse_;
  uint8_t* cuda_task_skip_default_bin_;
  uint8_t* cuda_task_na_as_missing_;
  uint8_t* cuda_task_out_default_left_;

  // CUDA memory, held by other object
  const hist_t* cuda_hist_;
  const int* cuda_num_features_;
};

}

#endif  // USE_CUDA
#endif  // LIGHTGBM_CUDA_HISTOGRAM_CONSTRUCTOR_HPP_
