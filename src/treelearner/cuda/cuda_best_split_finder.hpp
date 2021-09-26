/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifndef LIGHTGBM_CUDA_BEST_SPLIT_FINDER_HPP_
#define LIGHTGBM_CUDA_BEST_SPLIT_FINDER_HPP_

#ifdef USE_CUDA

#include "cuda_leaf_splits.hpp"

#include <LightGBM/bin.h>
#include <LightGBM/cuda/cuda_split_info.hpp>
#include <LightGBM/dataset.h>

#include <vector>

#define MAX_NUM_BIN_IN_FEATURE (256)
#define NUM_THREADS_FIND_BEST_LEAF (256)
#define NUM_TASKS_PER_SYNC_BLOCK (1024)

namespace LightGBM {

class CUDABestSplitFinder {
 public:
  CUDABestSplitFinder(
    const hist_t* cuda_hist,
    const Dataset* train_data,
    const std::vector<uint32_t>& feature_hist_offsets,
    const Config* config);

  ~CUDABestSplitFinder();

  void InitFeatureMetaInfo(const Dataset* train_data);

  void Init();

  void InitCUDAFeatureMetaInfo();

  void BeforeTrain(const std::vector<int8_t>& is_feature_used_bytree);

  void FindBestSplitsForLeaf(
    const CUDALeafSplitsStruct* smaller_leaf_splits,
    const CUDALeafSplitsStruct* larger_leaf_splits,
    const int smaller_leaf_index,
    const int larger_leaf_index,
    const data_size_t num_data_in_smaller_leaf,
    const data_size_t num_data_in_larger_leaf,
    const double sum_hessians_in_smaller_leaf,
    const double sum_hessians_in_larger_leaf);

  const CUDASplitInfo* FindBestFromAllSplits(
    const int cur_num_leaves,
    const int smaller_leaf_index,
    const int larger_leaf_index,
    int* smaller_leaf_best_split_feature,
    uint32_t* smaller_leaf_best_split_threshold,
    uint8_t* smaller_leaf_best_split_default_left,
    int* larger_leaf_best_split_feature,
    uint32_t* larger_leaf_best_split_threshold,
    uint8_t* larger_leaf_best_split_default_left,
    int* best_leaf_index);

  void ResetTrainingData(
    const hist_t* cuda_hist,
    const Dataset* train_data,
    const std::vector<uint32_t>& feature_hist_offsets);

  void ResetConfig(const Config* config);

 private:
  void LaunchFindBestSplitsForLeafKernel(const CUDALeafSplitsStruct* smaller_leaf_splits,
    const CUDALeafSplitsStruct* larger_leaf_splits, const int smaller_leaf_index, const int larger_leaf_index,
    const bool is_smaller_leaf_valid, const bool is_larger_leaf_valid);

  void LaunchSyncBestSplitForLeafKernel(
    const int host_smaller_leaf_index,
    const int host_larger_leaf_index,
    const bool is_smaller_leaf_valid,
    const bool is_larger_leaf_valid);

  void LaunchFindBestFromAllSplitsKernel(const int cur_num_leaves, const int smaller_leaf_index,
    const int larger_leaf_index,
    int* smaller_leaf_best_split_feature,
    uint32_t* smaller_leaf_best_split_threshold,
    uint8_t* smaller_leaf_best_split_default_left,
    int* larger_leaf_best_split_feature,
    uint32_t* larger_leaf_best_split_threshold,
    uint8_t* larger_leaf_best_split_default_left,
    int* best_leaf_index);

  // Host memory
  int num_features_;
  int num_leaves_;
  int max_num_bin_in_feature_;
  std::vector<uint32_t> feature_hist_offsets_;
  std::vector<uint8_t> feature_mfb_offsets_;
  std::vector<uint32_t> feature_default_bins_;
  std::vector<uint32_t> feature_num_bins_;
  std::vector<MissingType> feature_missing_type_;
  double lambda_l1_;
  double lambda_l2_;
  data_size_t min_data_in_leaf_;
  double min_sum_hessian_in_leaf_;
  double min_gain_to_split_;
  std::vector<cudaStream_t> cuda_streams_;
  // for best split find tasks
  std::vector<int> host_task_feature_index_;
  std::vector<uint8_t> host_task_reverse_;
  std::vector<uint8_t> host_task_skip_default_bin_;
  std::vector<uint8_t> host_task_na_as_missing_;
  std::vector<uint8_t> host_task_out_default_left_;
  int num_tasks_;

  // CUDA memory, held by this object
  // for per leaf best split information
  CUDASplitInfo* cuda_leaf_best_split_info_;
  // for best split information when finding best split
  CUDASplitInfo* cuda_best_split_info_;
  // feature information
  uint32_t* cuda_feature_hist_offsets_;
  uint8_t* cuda_feature_mfb_offsets_;
  uint32_t* cuda_feature_default_bins_;
  uint32_t* cuda_feature_num_bins_;
  // best split information buffer, to be copied to host
  int* cuda_best_split_info_buffer_;
  // find best split task information
  int* cuda_task_feature_index_;
  uint8_t* cuda_task_reverse_;
  uint8_t* cuda_task_skip_default_bin_;
  uint8_t* cuda_task_na_as_missing_;
  uint8_t* cuda_task_out_default_left_;
  int8_t* cuda_is_feature_used_bytree_;

  // CUDA memory, held by other object
  const hist_t* cuda_hist_;
};

}

#endif  // USE_CUDA
#endif  // LIGHTGBM_CUDA_HISTOGRAM_CONSTRUCTOR_HPP_
