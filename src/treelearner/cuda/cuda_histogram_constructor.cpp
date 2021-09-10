/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_histogram_constructor.hpp"

namespace LightGBM {

CUDAHistogramConstructor::CUDAHistogramConstructor(
  const Dataset* train_data,
  const int num_leaves,
  const int num_threads,
  const std::vector<uint32_t>& feature_hist_offsets,
  const int min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const int gpu_device_id):
  num_data_(train_data->num_data()),
  num_features_(train_data->num_features()),
  num_leaves_(num_leaves),
  num_threads_(num_threads),
  num_feature_groups_(train_data->num_feature_groups()),
  min_data_in_leaf_(min_data_in_leaf),
  min_sum_hessian_in_leaf_(min_sum_hessian_in_leaf),
  gpu_device_id_(gpu_device_id) {
  int offset = 0;
  for (int group_id = 0; group_id < train_data->num_feature_groups(); ++group_id) {
    offset += train_data->FeatureGroupNumBin(group_id);
  }
  need_fix_histogram_features_.clear();
  need_fix_histogram_features_num_bin_aligend_.clear();
  for (int feature_index = 0; feature_index < train_data->num_features(); ++feature_index) {
    const BinMapper* bin_mapper = train_data->FeatureBinMapper(feature_index);
    const uint32_t most_freq_bin = bin_mapper->GetMostFreqBin();
    if (most_freq_bin != 0) {
      need_fix_histogram_features_.emplace_back(feature_index);
      uint32_t num_bin_ref = static_cast<uint32_t>(bin_mapper->num_bin()) - 1;
      uint32_t num_bin_aligned = 1;
      while (num_bin_ref > 0) {
        num_bin_aligned <<= 1;
        num_bin_ref >>= 1;
      }
      need_fix_histogram_features_num_bin_aligend_.emplace_back(num_bin_aligned);
    }
    feature_num_bins_.emplace_back(static_cast<uint32_t>(bin_mapper->num_bin()));
    feature_most_freq_bins_.emplace_back(most_freq_bin);
  }
  feature_hist_offsets_.clear();
  for (size_t i = 0; i < feature_hist_offsets.size(); ++i) {
    feature_hist_offsets_.emplace_back(feature_hist_offsets[i]);
  }
  num_total_bin_ = offset;
  cuda_row_data_.reset(nullptr);
}

void CUDAHistogramConstructor::BeforeTrain(const score_t* gradients, const score_t* hessians) {
  cuda_gradients_ = gradients;
  cuda_hessians_ = hessians;
  SetCUDAMemoryOuter<hist_t>(cuda_hist_, 0, num_total_bin_ * 2 * num_leaves_, __FILE__, __LINE__);
}

void CUDAHistogramConstructor::Init(const Dataset* train_data, TrainingShareStates* share_state) {
  AllocateCUDAMemoryOuter<hist_t>(&cuda_hist_, num_total_bin_ * 2 * num_leaves_, __FILE__, __LINE__);
  SetCUDAMemoryOuter<hist_t>(cuda_hist_, 0, num_total_bin_ * 2 * num_leaves_, __FILE__, __LINE__);

  InitCUDAMemoryFromHostMemoryOuter<uint32_t>(&cuda_feature_num_bins_,
    feature_num_bins_.data(), feature_num_bins_.size(), __FILE__, __LINE__);

  InitCUDAMemoryFromHostMemoryOuter<uint32_t>(&cuda_feature_hist_offsets_,
    feature_hist_offsets_.data(), feature_hist_offsets_.size(), __FILE__, __LINE__);

  InitCUDAMemoryFromHostMemoryOuter<uint32_t>(&cuda_feature_most_freq_bins_,
    feature_most_freq_bins_.data(), feature_most_freq_bins_.size(), __FILE__, __LINE__);

  cuda_row_data_.reset(new CUDARowData(train_data, share_state, gpu_device_id_));
  cuda_row_data_->Init(train_data, share_state);

  CUDASUCCESS_OR_FATAL(cudaStreamCreate(&cuda_stream_));

  InitCUDAMemoryFromHostMemoryOuter<int>(&cuda_need_fix_histogram_features_, need_fix_histogram_features_.data(), need_fix_histogram_features_.size(), __FILE__, __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<uint32_t>(&cuda_need_fix_histogram_features_num_bin_aligned_, need_fix_histogram_features_num_bin_aligend_.data(),
    need_fix_histogram_features_num_bin_aligend_.size(), __FILE__, __LINE__);
}

void CUDAHistogramConstructor::ConstructHistogramForLeaf(
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const CUDALeafSplitsStruct* cuda_larger_leaf_splits, 
  const data_size_t num_data_in_smaller_leaf,
  const data_size_t num_data_in_larger_leaf,
  const double sum_hessians_in_smaller_leaf,
  const double sum_hessians_in_larger_leaf) {
  if ((num_data_in_smaller_leaf <= min_data_in_leaf_ || sum_hessians_in_smaller_leaf <= min_sum_hessian_in_leaf_) &&
    (num_data_in_larger_leaf <= min_data_in_leaf_ || sum_hessians_in_larger_leaf <= min_sum_hessian_in_leaf_)) {
    return;
  }
  LaunchConstructHistogramKernel(cuda_smaller_leaf_splits, num_data_in_smaller_leaf);
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
  global_timer.Start("CUDAHistogramConstructor::ConstructHistogramForLeaf::LaunchSubtractHistogramKernel");
  LaunchSubtractHistogramKernel(cuda_smaller_leaf_splits, cuda_larger_leaf_splits);
  global_timer.Stop("CUDAHistogramConstructor::ConstructHistogramForLeaf::LaunchSubtractHistogramKernel");
}

void CUDAHistogramConstructor::CalcConstructHistogramKernelDim(
  int* grid_dim_x,
  int* grid_dim_y,
  int* block_dim_x,
  int* block_dim_y,
  const data_size_t num_data_in_smaller_leaf) {
  *block_dim_x = cuda_row_data_->max_num_column_per_partition();
  *block_dim_y = NUM_THRADS_PER_BLOCK / cuda_row_data_->max_num_column_per_partition();
  *grid_dim_x = cuda_row_data_->num_feature_partitions();
  *grid_dim_y = std::max(min_grid_dim_y_,
    ((num_data_in_smaller_leaf + NUM_DATA_PER_THREAD - 1) / NUM_DATA_PER_THREAD + (*block_dim_y) - 1) / (*block_dim_y));
}

}  // namespace LightGBM

#endif  // USE_CUDA
