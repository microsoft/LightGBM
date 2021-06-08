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
  AllocateCUDAMemory<int>(1, &cuda_best_leaf_);
  AllocateCUDAMemory<int>(static_cast<size_t>(num_leaves_), &cuda_leaf_best_split_feature_);
  AllocateCUDAMemory<uint8_t>(static_cast<size_t>(num_leaves_), &cuda_leaf_best_split_default_left_);
  AllocateCUDAMemory<uint32_t>(static_cast<size_t>(num_leaves_), &cuda_leaf_best_split_threshold_);
  AllocateCUDAMemory<double>(static_cast<size_t>(num_leaves_), &cuda_leaf_best_split_gain_);
  AllocateCUDAMemory<double>(static_cast<size_t>(num_leaves_), &cuda_leaf_best_split_left_sum_gradient_);
  AllocateCUDAMemory<double>(static_cast<size_t>(num_leaves_), &cuda_leaf_best_split_left_sum_hessian_);
  AllocateCUDAMemory<data_size_t>(static_cast<size_t>(num_leaves_), &cuda_leaf_best_split_left_count_);
  AllocateCUDAMemory<double>(static_cast<size_t>(num_leaves_), &cuda_leaf_best_split_left_gain_);
  AllocateCUDAMemory<double>(static_cast<size_t>(num_leaves_), &cuda_leaf_best_split_left_output_);
  AllocateCUDAMemory<double>(static_cast<size_t>(num_leaves_), &cuda_leaf_best_split_right_sum_gradient_);
  AllocateCUDAMemory<double>(static_cast<size_t>(num_leaves_), &cuda_leaf_best_split_right_sum_hessian_);
  AllocateCUDAMemory<data_size_t>(static_cast<size_t>(num_leaves_), &cuda_leaf_best_split_right_count_);
  AllocateCUDAMemory<double>(static_cast<size_t>(num_leaves_), &cuda_leaf_best_split_right_gain_);
  AllocateCUDAMemory<double>(static_cast<size_t>(num_leaves_), &cuda_leaf_best_split_right_output_);
  AllocateCUDAMemory<uint8_t>(static_cast<size_t>(num_leaves_), &cuda_leaf_best_split_found_);

  AllocateCUDAMemory<uint32_t>(feature_hist_offsets_.size() * 2, &cuda_feature_hist_offsets_);
  CopyFromHostToCUDADevice<uint32_t>(cuda_feature_hist_offsets_, feature_hist_offsets_.data(), feature_hist_offsets_.size());

  AllocateCUDAMemory<uint8_t>(feature_mfb_offsets_.size(), &cuda_feature_mfb_offsets_);
  CopyFromHostToCUDADevice<uint8_t>(cuda_feature_mfb_offsets_, feature_mfb_offsets_.data(), feature_mfb_offsets_.size());

  AllocateCUDAMemory<uint32_t>(feature_default_bins_.size(), &cuda_feature_default_bins_);
  CopyFromHostToCUDADevice<uint32_t>(cuda_feature_default_bins_, feature_default_bins_.data(), feature_default_bins_.size());

  AllocateCUDAMemory<int>(1, &cuda_num_total_bin_);
  CopyFromHostToCUDADevice<int>(cuda_num_total_bin_, &num_total_bin_, 1);

  AllocateCUDAMemory<uint8_t>(num_features_, &cuda_feature_missing_type_);
  CopyFromHostToCUDADevice<uint8_t>(cuda_feature_missing_type_, feature_missing_type_.data(), static_cast<size_t>(num_features_));

  InitCUDAMemoryFromHostMemory<uint32_t>(&cuda_feature_num_bins_, feature_num_bins_.data(), static_cast<size_t>(num_features_));

  AllocateCUDAMemory<hist_t>(num_total_bin_ * 2, &prefix_sum_hist_left_);
  AllocateCUDAMemory<hist_t>(num_total_bin_ * 2, &prefix_sum_hist_right_);

  num_tasks_ = 0;
  for (int inner_feature_index = 0; inner_feature_index < num_features_; ++inner_feature_index) {
    const uint32_t num_bin = feature_num_bins_[inner_feature_index];
    const uint8_t missing_type = feature_missing_type_[inner_feature_index];
    if (num_bin > 2 && missing_type != 0) {
      if (missing_type == 1) {
        cpu_task_reverse_.emplace_back(0);
        cpu_task_reverse_.emplace_back(1);
        cpu_task_skip_default_bin_.emplace_back(1);
        cpu_task_skip_default_bin_.emplace_back(1);
        cpu_task_na_as_missing_.emplace_back(0);
        cpu_task_na_as_missing_.emplace_back(0);
        cpu_task_feature_index_.emplace_back(inner_feature_index);
        cpu_task_feature_index_.emplace_back(inner_feature_index);
        cpu_task_out_default_left_.emplace_back(0);
        cpu_task_out_default_left_.emplace_back(1);
        num_tasks_ += 2;
      } else {
        cpu_task_reverse_.emplace_back(0);
        cpu_task_reverse_.emplace_back(1);
        cpu_task_skip_default_bin_.emplace_back(0);
        cpu_task_skip_default_bin_.emplace_back(0);
        cpu_task_na_as_missing_.emplace_back(1);
        cpu_task_na_as_missing_.emplace_back(1);
        cpu_task_feature_index_.emplace_back(inner_feature_index);
        cpu_task_feature_index_.emplace_back(inner_feature_index);
        cpu_task_out_default_left_.emplace_back(0);
        cpu_task_out_default_left_.emplace_back(1);
        num_tasks_ += 2;
      }
    } else {
      cpu_task_reverse_.emplace_back(1);
      cpu_task_skip_default_bin_.emplace_back(0);
      cpu_task_na_as_missing_.emplace_back(0);
      cpu_task_feature_index_.emplace_back(inner_feature_index);
      if (missing_type != 2) {
        cpu_task_out_default_left_.emplace_back(1);
      } else {
        cpu_task_out_default_left_.emplace_back(0);
      }
      ++num_tasks_;
    }
  }
  InitCUDAMemoryFromHostMemory<int>(&cuda_task_feature_index_, cpu_task_feature_index_.data(), cpu_task_feature_index_.size());
  InitCUDAMemoryFromHostMemory<uint8_t>(&cuda_task_reverse_, cpu_task_reverse_.data(), cpu_task_reverse_.size());
  InitCUDAMemoryFromHostMemory<uint8_t>(&cuda_task_skip_default_bin_, cpu_task_skip_default_bin_.data(), cpu_task_skip_default_bin_.size());
  InitCUDAMemoryFromHostMemory<uint8_t>(&cuda_task_na_as_missing_, cpu_task_na_as_missing_.data(), cpu_task_na_as_missing_.size());
  InitCUDAMemoryFromHostMemory<uint8_t>(&cuda_task_out_default_left_, cpu_task_out_default_left_.data(), cpu_task_out_default_left_.size());

  const size_t output_buffer_size = 2 * static_cast<size_t>(num_tasks_);
  AllocateCUDAMemory<uint8_t>(output_buffer_size, &cuda_best_split_default_left_);
  AllocateCUDAMemory<uint32_t>(output_buffer_size, &cuda_best_split_threshold_);
  AllocateCUDAMemory<double>(output_buffer_size, &cuda_best_split_gain_);
  AllocateCUDAMemory<double>(output_buffer_size, &cuda_best_split_left_sum_gradient_);
  AllocateCUDAMemory<double>(output_buffer_size, &cuda_best_split_left_sum_hessian_);
  AllocateCUDAMemory<data_size_t>(output_buffer_size, &cuda_best_split_left_count_);
  AllocateCUDAMemory<double>(output_buffer_size, &cuda_best_split_left_gain_);
  AllocateCUDAMemory<double>(output_buffer_size, &cuda_best_split_left_output_);
  AllocateCUDAMemory<double>(output_buffer_size, &cuda_best_split_right_sum_gradient_);
  AllocateCUDAMemory<double>(output_buffer_size, &cuda_best_split_right_sum_hessian_);
  AllocateCUDAMemory<data_size_t>(output_buffer_size, &cuda_best_split_right_count_);
  AllocateCUDAMemory<double>(output_buffer_size, &cuda_best_split_right_gain_);
  AllocateCUDAMemory<double>(output_buffer_size, &cuda_best_split_right_output_);
  AllocateCUDAMemory<uint8_t>(output_buffer_size, &cuda_best_split_found_);

  AllocateCUDAMemory<int>(7, &cuda_best_split_info_buffer_);
  cuda_streams_.resize(2);
  CUDASUCCESS_OR_FATAL(cudaStreamCreate(&cuda_streams_[0]));
  CUDASUCCESS_OR_FATAL(cudaStreamCreate(&cuda_streams_[1]));
}

void CUDABestSplitFinder::BeforeTrain() {
  SetCUDAMemory<double>(cuda_leaf_best_split_gain_, 0, static_cast<size_t>(num_leaves_));
  SetCUDAMemory<uint8_t>(cuda_best_split_found_, 0, static_cast<size_t>(num_tasks_));
  SetCUDAMemory<double>(cuda_best_split_gain_, 0,  static_cast<size_t>(num_tasks_));
  SetCUDAMemory<uint8_t>(cuda_leaf_best_split_found_, 0, static_cast<size_t>(num_leaves_));
}

void CUDABestSplitFinder::FindBestSplitsForLeaf(const CUDALeafSplits* smaller_leaf_splits,
  const CUDALeafSplits* larger_leaf_splits, const int smaller_leaf_index, const int larger_leaf_index,
  const data_size_t num_data_in_smaller_leaf, const data_size_t num_data_in_larger_leaf,
  const double sum_hessians_in_smaller_leaf, const double sum_hessians_in_larger_leaf) {
  auto start = std::chrono::steady_clock::now();
  const bool is_smaller_leaf_valid = (num_data_in_smaller_leaf > min_data_in_leaf_ && sum_hessians_in_smaller_leaf > min_sum_hessian_in_leaf_);
  const bool is_larger_leaf_valid = (num_data_in_larger_leaf > min_data_in_leaf_ && sum_hessians_in_larger_leaf > min_sum_hessian_in_leaf_);
  LaunchFindBestSplitsForLeafKernel(smaller_leaf_splits, larger_leaf_splits,
    smaller_leaf_index, larger_leaf_index, is_smaller_leaf_valid, is_larger_leaf_valid);
  SynchronizeCUDADevice();
  global_timer.Start("CUDABestSplitFinder::LaunchSyncBestSplitForLeafKernel");
  LaunchSyncBestSplitForLeafKernel(smaller_leaf_index, larger_leaf_index, is_smaller_leaf_valid, is_larger_leaf_valid);
  SynchronizeCUDADevice();
  global_timer.Stop("CUDABestSplitFinder::LaunchSyncBestSplitForLeafKernel");
  auto end = std::chrono::steady_clock::now();
  double duration = (static_cast<std::chrono::duration<double>>(end - start)).count();
  //Log::Warning("FindBestSplitsForLeaf time %f", duration);
}

void CUDABestSplitFinder::FindBestFromAllSplits(const int* cuda_cur_num_leaves, const int smaller_leaf_index,
    const int larger_leaf_index, std::vector<int>* leaf_best_split_feature,
    std::vector<uint32_t>* leaf_best_split_threshold, std::vector<uint8_t>* leaf_best_split_default_left, int* best_leaf_index) {
  auto start = std::chrono::steady_clock::now();
  LaunchFindBestFromAllSplitsKernel(cuda_cur_num_leaves, smaller_leaf_index, larger_leaf_index,
    leaf_best_split_feature, leaf_best_split_threshold, leaf_best_split_default_left, best_leaf_index);
  SynchronizeCUDADevice();
  auto end = std::chrono::steady_clock::now();
  double duration = (static_cast<std::chrono::duration<double>>(end - start)).count();
}

}  // namespace LightGBM

#endif  // USE_CUDA
