/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_best_split_finder.hpp"
#include "cuda_leaf_splits.hpp"

namespace LightGBM {

CUDABestSplitFinder::CUDABestSplitFinder(const hist_t* cuda_hist, const Dataset* train_data,
  const std::vector<uint32_t>& feature_hist_offsets, const int num_leaves,
  const double lambda_l1, const double lambda_l2, const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf, const double min_gain_to_split,
  const int* cuda_num_features):
  num_features_(train_data->num_features()), num_leaves_(num_leaves),
  num_total_bin_(feature_hist_offsets.back()), feature_hist_offsets_(feature_hist_offsets), lambda_l1_(lambda_l1), lambda_l2_(lambda_l2),
  min_data_in_leaf_(min_data_in_leaf), min_sum_hessian_in_leaf_(min_sum_hessian_in_leaf), min_gain_to_split_(min_gain_to_split),
  cuda_hist_(cuda_hist), cuda_num_features_(cuda_num_features) {
  feature_missing_type_.resize(num_features_);
  feature_mfb_offsets_.resize(num_features_);
  feature_default_bins_.resize(num_features_);
  max_num_bin_in_feature_ = 0;
  for (int inner_feature_index = 0; inner_feature_index < num_features_; ++inner_feature_index) {
    const BinMapper* bin_mapper = train_data->FeatureBinMapper(inner_feature_index);
    const MissingType missing_type = bin_mapper->missing_type();
    feature_missing_type_[inner_feature_index] = missing_type;
    feature_mfb_offsets_[inner_feature_index] = static_cast<int8_t>(bin_mapper->GetMostFreqBin() == 0);
    feature_default_bins_[inner_feature_index] = bin_mapper->GetDefaultBin();
    const int num_bin = bin_mapper->num_bin() - feature_mfb_offsets_[inner_feature_index];
    if (num_bin > max_num_bin_in_feature_) {
      max_num_bin_in_feature_ = num_bin;
    }
  }
  if (max_num_bin_in_feature_ > MAX_NUM_BIN_IN_FEATURE) {
    Log::Fatal("feature bin size %d exceeds limit %d", max_num_bin_in_feature_, MAX_NUM_BIN_IN_FEATURE);
  }
}

void CUDABestSplitFinder::Init() {
  AllocateCUDAMemory<int>(1, &cuda_best_leaf_);
  AllocateCUDAMemory<int>(static_cast<size_t>(num_leaves_), &cuda_leaf_best_split_feature_);
  AllocateCUDAMemory<uint8_t>(static_cast<size_t>(num_leaves_), &cuda_leaf_best_split_default_left_);
  AllocateCUDAMemory<uint32_t>(static_cast<size_t>(num_leaves_), &cuda_leaf_best_split_threshold_);
  AllocateCUDAMemory<double>(static_cast<size_t>(num_leaves_), &cuda_leaf_best_split_gain_);
  AllocateCUDAMemory<double>(static_cast<size_t>(num_leaves_), &cuda_leaf_best_split_left_sum_gradient_);
  AllocateCUDAMemory<double>(static_cast<size_t>(num_leaves_), &cuda_leaf_best_split_left_sum_hessian_);
  AllocateCUDAMemory<data_size_t>(static_cast<size_t>(num_leaves_), &cuda_leaf_best_split_left_count_);
  AllocateCUDAMemory<double>(static_cast<size_t>(num_leaves_), &cuda_leaf_best_split_left_output_);
  AllocateCUDAMemory<double>(static_cast<size_t>(num_leaves_), &cuda_leaf_best_split_right_sum_gradient_);
  AllocateCUDAMemory<double>(static_cast<size_t>(num_leaves_), &cuda_leaf_best_split_right_sum_hessian_);
  AllocateCUDAMemory<data_size_t>(static_cast<size_t>(num_leaves_), &cuda_leaf_best_split_right_count_);
  AllocateCUDAMemory<double>(static_cast<size_t>(num_leaves_), &cuda_leaf_best_split_right_output_);

  AllocateCUDAMemory<uint32_t>(feature_hist_offsets_.size(), &cuda_feature_hist_offsets_);
  CopyFromHostToCUDADevice<uint32_t>(cuda_feature_hist_offsets_, feature_hist_offsets_.data(), feature_hist_offsets_.size());

  AllocateCUDAMemory<uint8_t>(feature_mfb_offsets_.size(), &cuda_feature_mfb_offsets_);
  CopyFromHostToCUDADevice<uint8_t>(cuda_feature_mfb_offsets_, feature_mfb_offsets_.data(), feature_mfb_offsets_.size());

  AllocateCUDAMemory<uint32_t>(feature_default_bins_.size(), &cuda_feature_default_bins_);
  CopyFromHostToCUDADevice<uint32_t>(cuda_feature_default_bins_, feature_default_bins_.data(), feature_default_bins_.size());

  AllocateCUDAMemory<int>(1, &cuda_num_total_bin_);
  CopyFromHostToCUDADevice<int>(cuda_num_total_bin_, &num_total_bin_, 1);

  AllocateCUDAMemory<uint8_t>(num_features_, &cuda_feature_missing_type_);
  CopyFromHostToCUDADevice<uint8_t>(cuda_feature_missing_type_, feature_missing_type_.data(), static_cast<size_t>(num_features_));

  AllocateCUDAMemory<double>(1, &cuda_lambda_l1_);
  CopyFromHostToCUDADevice<double>(cuda_lambda_l1_, &lambda_l1_, 1);

  InitCUDAMemoryFromHostMemory<double>(&cuda_lambda_l2_, &lambda_l2_, 1);

  AllocateCUDAMemory<hist_t>(num_total_bin_ * 2, &prefix_sum_hist_left_);
  AllocateCUDAMemory<hist_t>(num_total_bin_ * 2, &prefix_sum_hist_right_);

  // * 2 for smaller and larger leaves, * 2 for default left or not
  const size_t feature_best_split_info_buffer_size = static_cast<size_t>(num_features_) * 4;
  AllocateCUDAMemory<int>(feature_best_split_info_buffer_size, &cuda_best_split_feature_);
  AllocateCUDAMemory<uint8_t>(feature_best_split_info_buffer_size, &cuda_best_split_default_left_);
  AllocateCUDAMemory<uint32_t>(feature_best_split_info_buffer_size, &cuda_best_split_threshold_);
  AllocateCUDAMemory<double>(feature_best_split_info_buffer_size, &cuda_best_split_gain_);
  AllocateCUDAMemory<double>(feature_best_split_info_buffer_size, &cuda_best_split_left_sum_gradient_);
  AllocateCUDAMemory<double>(feature_best_split_info_buffer_size, &cuda_best_split_left_sum_hessian_);
  AllocateCUDAMemory<data_size_t>(feature_best_split_info_buffer_size, &cuda_best_split_left_count_);
  AllocateCUDAMemory<double>(feature_best_split_info_buffer_size, &cuda_best_split_left_output_);
  AllocateCUDAMemory<double>(feature_best_split_info_buffer_size, &cuda_best_split_right_sum_gradient_);
  AllocateCUDAMemory<double>(feature_best_split_info_buffer_size, &cuda_best_split_right_sum_hessian_);
  AllocateCUDAMemory<data_size_t>(feature_best_split_info_buffer_size, &cuda_best_split_right_count_);
  AllocateCUDAMemory<double>(feature_best_split_info_buffer_size, &cuda_best_split_right_output_);
  AllocateCUDAMemory<uint8_t>(feature_best_split_info_buffer_size, &cuda_best_split_found_);

  AllocateCUDAMemory<data_size_t>(1, &cuda_min_data_in_leaf_);
  CopyFromHostToCUDADevice<data_size_t>(cuda_min_data_in_leaf_, &min_data_in_leaf_, 1);
  AllocateCUDAMemory<double>(1, &cuda_min_sum_hessian_in_leaf_);
  CopyFromHostToCUDADevice<double>(cuda_min_sum_hessian_in_leaf_, &min_sum_hessian_in_leaf_, 1);
  AllocateCUDAMemory<double>(1, &cuda_min_gain_to_split_);
  CopyFromHostToCUDADevice<double>(cuda_min_gain_to_split_, &min_gain_to_split_, 1);
}

void CUDABestSplitFinder::FindBestSplitsForLeaf(const CUDALeafSplits* smaller_leaf_splits,
  const CUDALeafSplits* larger_leaf_splits) {
  auto start = std::chrono::steady_clock::now();
  LaunchFindBestSplitsForLeafKernel(smaller_leaf_splits->cuda_leaf_index(),
    larger_leaf_splits->cuda_leaf_index(),
    smaller_leaf_splits->cuda_gain(),
    larger_leaf_splits->cuda_gain(),
    smaller_leaf_splits->cuda_sum_of_gradients(),
    smaller_leaf_splits->cuda_sum_of_hessians(),
    smaller_leaf_splits->cuda_num_data_in_leaf(),
    larger_leaf_splits->cuda_sum_of_gradients(),
    larger_leaf_splits->cuda_sum_of_hessians(),
    larger_leaf_splits->cuda_num_data_in_leaf());
  SynchronizeCUDADevice();
  LaunchSyncBestSplitForLeafKernel(smaller_leaf_splits->cuda_leaf_index(), larger_leaf_splits->cuda_leaf_index());
  SynchronizeCUDADevice();
  auto end = std::chrono::steady_clock::now();
  double duration = (static_cast<std::chrono::duration<double>>(end - start)).count();
  Log::Warning("FindBestSplitsForLeaf time %f", duration);
}

void CUDABestSplitFinder::FindBestFromAllSplits(const int* cuda_cur_num_leaves) {
  auto start = std::chrono::steady_clock::now();
  LaunchFindBestFromAllSplitsKernel(cuda_cur_num_leaves);
  auto end = std::chrono::steady_clock::now();
  double duration = (static_cast<std::chrono::duration<double>>(end - start)).count();
  Log::Warning("FindBestFromAllSplits time %f", duration);
}

}  // namespace LightGBM

#endif  // USE_CUDA
