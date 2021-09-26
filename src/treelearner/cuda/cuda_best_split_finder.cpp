/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_best_split_finder.hpp"
#include "cuda_leaf_splits.hpp"

namespace LightGBM {

CUDABestSplitFinder::CUDABestSplitFinder(
  const hist_t* cuda_hist,
  const Dataset* train_data,
  const std::vector<uint32_t>& feature_hist_offsets,
  const Config* config):
  num_features_(train_data->num_features()),
  num_leaves_(config->num_leaves),
  feature_hist_offsets_(feature_hist_offsets),
  lambda_l1_(config->lambda_l1),
  lambda_l2_(config->lambda_l2),
  min_data_in_leaf_(config->min_data_in_leaf),
  min_sum_hessian_in_leaf_(config->min_sum_hessian_in_leaf),
  min_gain_to_split_(config->min_gain_to_split),
  cuda_hist_(cuda_hist) {
  InitFeatureMetaInfo(train_data);
  cuda_leaf_best_split_info_ = nullptr;
  cuda_best_split_info_ = nullptr;
  cuda_feature_hist_offsets_ = nullptr;
  cuda_feature_mfb_offsets_ = nullptr;
  cuda_feature_default_bins_ = nullptr;
  cuda_feature_num_bins_ = nullptr;
  cuda_best_split_info_buffer_ = nullptr;
  cuda_task_feature_index_ = nullptr;
  cuda_task_reverse_ = nullptr;
  cuda_task_skip_default_bin_ = nullptr;
  cuda_task_na_as_missing_ = nullptr;
  cuda_task_out_default_left_ = nullptr;
  cuda_is_feature_used_bytree_ = nullptr;
}

CUDABestSplitFinder::~CUDABestSplitFinder() {
  DeallocateCUDAMemory<CUDASplitInfo>(&cuda_leaf_best_split_info_, __FILE__, __LINE__);
  DeallocateCUDAMemory<CUDASplitInfo>(&cuda_best_split_info_, __FILE__, __LINE__);
  DeallocateCUDAMemory<uint32_t>(&cuda_feature_hist_offsets_, __FILE__, __LINE__);
  DeallocateCUDAMemory<uint8_t>(&cuda_feature_mfb_offsets_, __FILE__, __LINE__);
  DeallocateCUDAMemory<uint32_t>(&cuda_feature_default_bins_, __FILE__, __LINE__);
  DeallocateCUDAMemory<uint32_t>(&cuda_feature_num_bins_, __FILE__, __LINE__);
  DeallocateCUDAMemory<int>(&cuda_best_split_info_buffer_, __FILE__, __LINE__);
  DeallocateCUDAMemory<int>(&cuda_task_feature_index_, __FILE__, __LINE__);
  DeallocateCUDAMemory<uint8_t>(&cuda_task_reverse_, __FILE__, __LINE__);
  DeallocateCUDAMemory<uint8_t>(&cuda_task_skip_default_bin_, __FILE__, __LINE__);
  DeallocateCUDAMemory<uint8_t>(&cuda_task_na_as_missing_, __FILE__, __LINE__);
  DeallocateCUDAMemory<uint8_t>(&cuda_task_out_default_left_, __FILE__, __LINE__);
  DeallocateCUDAMemory<int8_t>(&cuda_is_feature_used_bytree_, __FILE__, __LINE__);
  gpuAssert(cudaStreamDestroy(cuda_streams_[0]), __FILE__, __LINE__);
  gpuAssert(cudaStreamDestroy(cuda_streams_[1]), __FILE__, __LINE__);
  cuda_streams_.clear();
  cuda_streams_.shrink_to_fit();
}

void CUDABestSplitFinder::InitFeatureMetaInfo(const Dataset* train_data) {
  feature_missing_type_.resize(num_features_);
  feature_mfb_offsets_.resize(num_features_);
  feature_default_bins_.resize(num_features_);
  feature_num_bins_.resize(num_features_);
  max_num_bin_in_feature_ = 0;
  for (int inner_feature_index = 0; inner_feature_index < num_features_; ++inner_feature_index) {
    const BinMapper* bin_mapper = train_data->FeatureBinMapper(inner_feature_index);
    const MissingType missing_type = bin_mapper->missing_type();
    feature_missing_type_[inner_feature_index] = missing_type;
    feature_mfb_offsets_[inner_feature_index] = static_cast<int8_t>(bin_mapper->GetMostFreqBin() == 0);
    feature_default_bins_[inner_feature_index] = bin_mapper->GetDefaultBin();
    feature_num_bins_[inner_feature_index] = static_cast<uint32_t>(bin_mapper->num_bin());
    const int num_bin_hist = bin_mapper->num_bin() - feature_mfb_offsets_[inner_feature_index];
    if (num_bin_hist > max_num_bin_in_feature_) {
      max_num_bin_in_feature_ = num_bin_hist;
    }
  }
  if (max_num_bin_in_feature_ > MAX_NUM_BIN_IN_FEATURE) {
    Log::Fatal("feature bin size %d exceeds limit %d", max_num_bin_in_feature_, MAX_NUM_BIN_IN_FEATURE);
  }
}

void CUDABestSplitFinder::Init() {
  InitCUDAFeatureMetaInfo();
  cuda_streams_.resize(2);
  CUDASUCCESS_OR_FATAL(cudaStreamCreate(&cuda_streams_[0]));
  CUDASUCCESS_OR_FATAL(cudaStreamCreate(&cuda_streams_[1]));
  AllocateCUDAMemory<int>(&cuda_best_split_info_buffer_, 7, __FILE__, __LINE__);
}

void CUDABestSplitFinder::InitCUDAFeatureMetaInfo() {
  InitCUDAMemoryFromHostMemory<uint32_t>(&cuda_feature_hist_offsets_,
                                          feature_hist_offsets_.data(),
                                          feature_hist_offsets_.size(),
                                          __FILE__,
                                          __LINE__);
  InitCUDAMemoryFromHostMemory<uint8_t>(&cuda_feature_mfb_offsets_,
                                         feature_mfb_offsets_.data(),
                                         feature_mfb_offsets_.size(),
                                         __FILE__,
                                         __LINE__);
  InitCUDAMemoryFromHostMemory<uint32_t>(&cuda_feature_default_bins_,
                                          feature_default_bins_.data(),
                                          feature_default_bins_.size(),
                                          __FILE__,
                                          __LINE__);
  InitCUDAMemoryFromHostMemory<uint32_t>(&cuda_feature_num_bins_,
                                              feature_num_bins_.data(),
                                              static_cast<size_t>(num_features_),
                                              __FILE__,
                                              __LINE__);
  AllocateCUDAMemory<int8_t>(&cuda_is_feature_used_bytree_, static_cast<size_t>(num_features_), __FILE__, __LINE__);
  num_tasks_ = 0;
  for (int inner_feature_index = 0; inner_feature_index < num_features_; ++inner_feature_index) {
    const uint32_t num_bin = feature_num_bins_[inner_feature_index];
    const uint8_t missing_type = feature_missing_type_[inner_feature_index];
    if (num_bin > 2 && missing_type != MissingType::None) {
      if (missing_type == MissingType::Zero) {
        host_task_reverse_.emplace_back(0);
        host_task_reverse_.emplace_back(1);
        host_task_skip_default_bin_.emplace_back(1);
        host_task_skip_default_bin_.emplace_back(1);
        host_task_na_as_missing_.emplace_back(0);
        host_task_na_as_missing_.emplace_back(0);
        host_task_feature_index_.emplace_back(inner_feature_index);
        host_task_feature_index_.emplace_back(inner_feature_index);
        host_task_out_default_left_.emplace_back(0);
        host_task_out_default_left_.emplace_back(1);
        num_tasks_ += 2;
      } else {
        host_task_reverse_.emplace_back(0);
        host_task_reverse_.emplace_back(1);
        host_task_skip_default_bin_.emplace_back(0);
        host_task_skip_default_bin_.emplace_back(0);
        host_task_na_as_missing_.emplace_back(1);
        host_task_na_as_missing_.emplace_back(1);
        host_task_feature_index_.emplace_back(inner_feature_index);
        host_task_feature_index_.emplace_back(inner_feature_index);
        host_task_out_default_left_.emplace_back(0);
        host_task_out_default_left_.emplace_back(1);
        num_tasks_ += 2;
      }
    } else {
      host_task_reverse_.emplace_back(1);
      host_task_skip_default_bin_.emplace_back(0);
      host_task_na_as_missing_.emplace_back(0);
      host_task_feature_index_.emplace_back(inner_feature_index);
      if (missing_type != 2) {
        host_task_out_default_left_.emplace_back(1);
      } else {
        host_task_out_default_left_.emplace_back(0);
      }
      ++num_tasks_;
    }
  }

  const int num_task_blocks = (num_tasks_ + NUM_TASKS_PER_SYNC_BLOCK - 1) / NUM_TASKS_PER_SYNC_BLOCK;
  const size_t cuda_best_leaf_split_info_buffer_size = static_cast<size_t>(num_task_blocks) * static_cast<size_t>(num_leaves_);

  AllocateCUDAMemory<CUDASplitInfo>(&cuda_leaf_best_split_info_,
                                         cuda_best_leaf_split_info_buffer_size,
                                         __FILE__,
                                         __LINE__);
  InitCUDAMemoryFromHostMemory<int>(&cuda_task_feature_index_,
                                         host_task_feature_index_.data(),
                                         host_task_feature_index_.size(),
                                         __FILE__,
                                         __LINE__);
  InitCUDAMemoryFromHostMemory<uint8_t>(&cuda_task_reverse_,
                                             host_task_reverse_.data(),
                                             host_task_reverse_.size(),
                                             __FILE__,
                                             __LINE__);
  InitCUDAMemoryFromHostMemory<uint8_t>(&cuda_task_skip_default_bin_,
                                             host_task_skip_default_bin_.data(),
                                             host_task_skip_default_bin_.size(),
                                             __FILE__,
                                             __LINE__);
  InitCUDAMemoryFromHostMemory<uint8_t>(&cuda_task_na_as_missing_,
                                             host_task_na_as_missing_.data(),
                                             host_task_na_as_missing_.size(),
                                             __FILE__,
                                             __LINE__);
  InitCUDAMemoryFromHostMemory<uint8_t>(&cuda_task_out_default_left_,
                                             host_task_out_default_left_.data(),
                                             host_task_out_default_left_.size(),
                                             __FILE__,
                                             __LINE__);

  const size_t output_buffer_size = 2 * static_cast<size_t>(num_tasks_);
  AllocateCUDAMemory<CUDASplitInfo>(&cuda_best_split_info_, output_buffer_size, __FILE__, __LINE__);
}

void CUDABestSplitFinder::ResetTrainingData(
  const hist_t* cuda_hist,
  const Dataset* train_data,
  const std::vector<uint32_t>& feature_hist_offsets) {
  cuda_hist_ = cuda_hist;
  num_features_ = train_data->num_features();
  feature_hist_offsets_ = feature_hist_offsets;
  InitFeatureMetaInfo(train_data);
  DeallocateCUDAMemory<uint32_t>(&cuda_feature_hist_offsets_, __FILE__, __LINE__);
  DeallocateCUDAMemory<uint32_t>(&cuda_feature_hist_offsets_, __FILE__, __LINE__);
  DeallocateCUDAMemory<uint8_t>(&cuda_feature_mfb_offsets_, __FILE__, __LINE__);
  DeallocateCUDAMemory<uint32_t>(&cuda_feature_default_bins_, __FILE__, __LINE__);
  DeallocateCUDAMemory<uint32_t>(&cuda_feature_num_bins_, __FILE__, __LINE__);
  DeallocateCUDAMemory<int8_t>(&cuda_is_feature_used_bytree_, __FILE__, __LINE__);
  DeallocateCUDAMemory<CUDASplitInfo>(&cuda_best_split_info_, __FILE__, __LINE__);
  host_task_reverse_.clear();
  host_task_skip_default_bin_.clear();
  host_task_na_as_missing_.clear();
  host_task_feature_index_.clear();
  host_task_out_default_left_.clear();
  InitCUDAFeatureMetaInfo();
}

void CUDABestSplitFinder::ResetConfig(const Config* config) {
  num_leaves_ = config->num_leaves;
  lambda_l1_ = config->lambda_l1;
  lambda_l2_ = config->lambda_l2;
  min_data_in_leaf_ = config->min_data_in_leaf;
  min_sum_hessian_in_leaf_ = config->min_sum_hessian_in_leaf;
  min_gain_to_split_ = config->min_gain_to_split;
  const int num_task_blocks = (num_tasks_ + NUM_TASKS_PER_SYNC_BLOCK - 1) / NUM_TASKS_PER_SYNC_BLOCK;
  const size_t cuda_best_leaf_split_info_buffer_size = static_cast<size_t>(num_task_blocks) * static_cast<size_t>(num_leaves_);
  DeallocateCUDAMemory<CUDASplitInfo>(&cuda_leaf_best_split_info_, __FILE__, __LINE__);
  AllocateCUDAMemory<CUDASplitInfo>(&cuda_leaf_best_split_info_,
                                         cuda_best_leaf_split_info_buffer_size,
                                         __FILE__,
                                         __LINE__);
}

void CUDABestSplitFinder::BeforeTrain(const std::vector<int8_t>& is_feature_used_bytree) {
  CopyFromHostToCUDADevice<int8_t>(cuda_is_feature_used_bytree_,
                                        is_feature_used_bytree.data(),
                                        is_feature_used_bytree.size(), __FILE__, __LINE__);
}

void CUDABestSplitFinder::FindBestSplitsForLeaf(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const CUDALeafSplitsStruct* larger_leaf_splits,
  const int smaller_leaf_index,
  const int larger_leaf_index,
  const data_size_t num_data_in_smaller_leaf,
  const data_size_t num_data_in_larger_leaf,
  const double sum_hessians_in_smaller_leaf,
  const double sum_hessians_in_larger_leaf) {
  const bool is_smaller_leaf_valid = (num_data_in_smaller_leaf > min_data_in_leaf_ &&
    sum_hessians_in_smaller_leaf > min_sum_hessian_in_leaf_);
  const bool is_larger_leaf_valid = (num_data_in_larger_leaf > min_data_in_leaf_ &&
    sum_hessians_in_larger_leaf > min_sum_hessian_in_leaf_ && larger_leaf_index >= 0);
  LaunchFindBestSplitsForLeafKernel(smaller_leaf_splits, larger_leaf_splits,
    smaller_leaf_index, larger_leaf_index, is_smaller_leaf_valid, is_larger_leaf_valid);
  global_timer.Start("CUDABestSplitFinder::LaunchSyncBestSplitForLeafKernel");
  LaunchSyncBestSplitForLeafKernel(smaller_leaf_index, larger_leaf_index, is_smaller_leaf_valid, is_larger_leaf_valid);
  SynchronizeCUDADevice(__FILE__, __LINE__);
  global_timer.Stop("CUDABestSplitFinder::LaunchSyncBestSplitForLeafKernel");
}

const CUDASplitInfo* CUDABestSplitFinder::FindBestFromAllSplits(
    const int cur_num_leaves,
    const int smaller_leaf_index,
    const int larger_leaf_index,
    int* smaller_leaf_best_split_feature,
    uint32_t* smaller_leaf_best_split_threshold,
    uint8_t* smaller_leaf_best_split_default_left,
    int* larger_leaf_best_split_feature,
    uint32_t* larger_leaf_best_split_threshold,
    uint8_t* larger_leaf_best_split_default_left,
    int* best_leaf_index) {
  LaunchFindBestFromAllSplitsKernel(
    cur_num_leaves,
    smaller_leaf_index,
    larger_leaf_index,
    smaller_leaf_best_split_feature,
    smaller_leaf_best_split_threshold,
    smaller_leaf_best_split_default_left,
    larger_leaf_best_split_feature,
    larger_leaf_best_split_threshold,
    larger_leaf_best_split_default_left,
    best_leaf_index);
  SynchronizeCUDADevice(__FILE__, __LINE__);
  return cuda_leaf_best_split_info_ + (*best_leaf_index);
}

}  // namespace LightGBM

#endif  // USE_CUDA
