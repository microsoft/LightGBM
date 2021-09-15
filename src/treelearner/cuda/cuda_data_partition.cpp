/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_data_partition.hpp"

namespace LightGBM {

CUDADataPartition::CUDADataPartition(
  const Dataset* train_data,
  const int num_total_bin,
  const int num_leaves,
  const int num_threads,
  hist_t* cuda_hist):

  num_data_(train_data->num_data()),
  num_features_(train_data->num_features()),
  num_total_bin_(num_total_bin),
  num_leaves_(num_leaves),
  num_threads_(num_threads),
  cuda_hist_(cuda_hist) {
  CalcBlockDim(num_data_);
  max_num_split_indices_blocks_ = grid_dim_;
  cur_num_leaves_ = 1;
  bin_upper_bounds_.resize(num_features_);
  feature_num_bins_.resize(num_features_);
  int cur_group = 0;
  uint32_t prev_group_bins = 0;
  for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
    const int group = train_data->Feature2Group(feature_index);
    if (cur_group != group) {
      prev_group_bins += static_cast<uint32_t>(train_data->FeatureGroupNumBin(cur_group));
      cur_group = group;
    }
    const BinMapper* bin_mapper = train_data->FeatureBinMapper(feature_index);
    bin_upper_bounds_[feature_index] = bin_mapper->bin_upper_bound();
    feature_num_bins_[feature_index] = bin_mapper->num_bin();
  }

  cuda_column_data_ = train_data->cuda_column_data();
}

void CUDADataPartition::Init() {
  // allocate CUDA memory
  AllocateCUDAMemoryOuter<data_size_t>(&cuda_data_indices_, static_cast<size_t>(num_data_), __FILE__, __LINE__);
  AllocateCUDAMemoryOuter<data_size_t>(&cuda_leaf_data_start_, static_cast<size_t>(num_leaves_), __FILE__, __LINE__);
  AllocateCUDAMemoryOuter<data_size_t>(&cuda_leaf_data_end_, static_cast<size_t>(num_leaves_), __FILE__, __LINE__);
  AllocateCUDAMemoryOuter<data_size_t>(&cuda_leaf_num_data_, static_cast<size_t>(num_leaves_), __FILE__, __LINE__);
  // leave some space for alignment
  AllocateCUDAMemoryOuter<uint16_t>(&cuda_block_to_left_offset_, static_cast<size_t>(num_data_), __FILE__, __LINE__);
  AllocateCUDAMemoryOuter<int>(&cuda_data_index_to_leaf_index_, static_cast<size_t>(num_data_), __FILE__, __LINE__);
  AllocateCUDAMemoryOuter<data_size_t>(&cuda_block_data_to_left_offset_, static_cast<size_t>(max_num_split_indices_blocks_) + 1, __FILE__, __LINE__);
  AllocateCUDAMemoryOuter<data_size_t>(&cuda_block_data_to_right_offset_, static_cast<size_t>(max_num_split_indices_blocks_) + 1, __FILE__, __LINE__);
  SetCUDAMemoryOuter<data_size_t>(cuda_block_data_to_left_offset_, 0, static_cast<size_t>(max_num_split_indices_blocks_) + 1, __FILE__, __LINE__);
  SetCUDAMemoryOuter<data_size_t>(cuda_block_data_to_right_offset_, 0, static_cast<size_t>(max_num_split_indices_blocks_) + 1, __FILE__, __LINE__);
  AllocateCUDAMemoryOuter<data_size_t>(&cuda_out_data_indices_in_leaf_, static_cast<size_t>(num_data_), __FILE__, __LINE__);
  AllocateCUDAMemoryOuter<hist_t*>(&cuda_hist_pool_, static_cast<size_t>(num_leaves_), __FILE__, __LINE__);
  CopyFromHostToCUDADeviceOuter<hist_t*>(cuda_hist_pool_, &cuda_hist_, 1, __FILE__, __LINE__);

  AllocateCUDAMemoryOuter<int>(&cuda_split_info_buffer_, 12, __FILE__, __LINE__);

  AllocateCUDAMemoryOuter<double>(&cuda_leaf_output_, static_cast<size_t>(num_leaves_), __FILE__, __LINE__);

  cuda_streams_.resize(4);
  CUDASUCCESS_OR_FATAL(cudaStreamCreate(&cuda_streams_[0]));
  CUDASUCCESS_OR_FATAL(cudaStreamCreate(&cuda_streams_[1]));
  CUDASUCCESS_OR_FATAL(cudaStreamCreate(&cuda_streams_[2]));
  CUDASUCCESS_OR_FATAL(cudaStreamCreate(&cuda_streams_[3]));

  std::vector<double> flatten_bin_upper_bounds;
  std::vector<int> feature_num_bin_offsets;
  int offset = 0;
  feature_num_bin_offsets.emplace_back(offset);
  for (size_t i = 0; i < bin_upper_bounds_.size(); ++i) {
    CHECK_EQ(static_cast<size_t>(feature_num_bins_[i]), bin_upper_bounds_[i].size());
    for (const auto value : bin_upper_bounds_[i]) {
      flatten_bin_upper_bounds.emplace_back(value);
    }
    offset += feature_num_bins_[i];
    feature_num_bin_offsets.emplace_back(offset);
  }
  InitCUDAMemoryFromHostMemoryOuter<int>(&cuda_feature_num_bin_offsets_, feature_num_bin_offsets.data(), feature_num_bin_offsets.size(), __FILE__, __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<double>(&cuda_bin_upper_bounds_, flatten_bin_upper_bounds.data(), flatten_bin_upper_bounds.size(), __FILE__, __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<data_size_t>(&cuda_num_data_, &num_data_, 1, __FILE__, __LINE__);
  use_bagging_ = false;
}

void CUDADataPartition::BeforeTrain() {
  if (!use_bagging_) {
    LaunchFillDataIndicesBeforeTrain();
  }
  SetCUDAMemoryOuter<data_size_t>(cuda_leaf_num_data_, 0, static_cast<size_t>(num_leaves_), __FILE__, __LINE__);
  SetCUDAMemoryOuter<data_size_t>(cuda_leaf_data_start_, 0, static_cast<size_t>(num_leaves_), __FILE__, __LINE__);
  SetCUDAMemoryOuter<data_size_t>(cuda_leaf_data_end_, 0, static_cast<size_t>(num_leaves_), __FILE__, __LINE__);
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
  if (!use_bagging_) {
    CopyFromCUDADeviceToCUDADeviceOuter<data_size_t>(cuda_leaf_num_data_, cuda_num_data_, 1, __FILE__, __LINE__);
    CopyFromCUDADeviceToCUDADeviceOuter<data_size_t>(cuda_leaf_data_end_, cuda_num_data_, 1, __FILE__, __LINE__);
  } else {
    CopyFromHostToCUDADeviceOuter<data_size_t>(cuda_leaf_num_data_, &num_used_indices_, 1, __FILE__, __LINE__);
    CopyFromHostToCUDADeviceOuter<data_size_t>(cuda_leaf_data_end_, &num_used_indices_, 1, __FILE__, __LINE__);
  }
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
  CopyFromHostToCUDADeviceOuter<hist_t*>(cuda_hist_pool_, &cuda_hist_, 1, __FILE__, __LINE__);
}

void CUDADataPartition::Split(
  // input best split info
  const CUDASplitInfo* best_split_info,
  const int left_leaf_index,
  const int right_leaf_index,
  const int leaf_best_split_feature,
  const uint32_t leaf_best_split_threshold,
  const uint8_t leaf_best_split_default_left,
  const data_size_t num_data_in_leaf,
  const data_size_t leaf_data_start,
  // for leaf information update
  CUDALeafSplitsStruct* smaller_leaf_splits,
  CUDALeafSplitsStruct* larger_leaf_splits,
  // gather information for CPU, used for launching kernels
  data_size_t* left_leaf_num_data,
  data_size_t* right_leaf_num_data,
  data_size_t* left_leaf_start,
  data_size_t* right_leaf_start,
  double* left_leaf_sum_of_hessians,
  double* right_leaf_sum_of_hessians) {
  CalcBlockDim(num_data_in_leaf);
  global_timer.Start("GenDataToLeftBitVector");
  GenDataToLeftBitVector(num_data_in_leaf,
                         leaf_best_split_feature,
                         leaf_best_split_threshold,
                         leaf_best_split_default_left,
                         leaf_data_start,
                         left_leaf_index,
                         right_leaf_index);
  global_timer.Stop("GenDataToLeftBitVector");
  global_timer.Start("SplitInner");

  SplitInner(num_data_in_leaf,
             best_split_info,
             left_leaf_index,
             right_leaf_index,
             smaller_leaf_splits,
             larger_leaf_splits,
             left_leaf_num_data,
             right_leaf_num_data,
             left_leaf_start,
             right_leaf_start,
             left_leaf_sum_of_hessians,
             right_leaf_sum_of_hessians);
  global_timer.Stop("SplitInner");
}

void CUDADataPartition::GenDataToLeftBitVector(
    const data_size_t num_data_in_leaf,
    const int split_feature_index,
    const uint32_t split_threshold,
    const uint8_t split_default_left,
    const data_size_t leaf_data_start,
    const int left_leaf_index,
    const int right_leaf_index) {
  LaunchGenDataToLeftBitVectorKernel(num_data_in_leaf,
                                     split_feature_index,
                                     split_threshold,
                                     split_default_left,
                                     leaf_data_start,
                                     left_leaf_index,
                                     right_leaf_index);
}

void CUDADataPartition::SplitInner(
  const data_size_t num_data_in_leaf,
  const CUDASplitInfo* best_split_info,
  const int left_leaf_index,
  const int right_leaf_index,
  // for leaf splits information update
  CUDALeafSplitsStruct* smaller_leaf_splits,
  CUDALeafSplitsStruct* larger_leaf_splits,
  data_size_t* left_leaf_num_data,
  data_size_t* right_leaf_num_data,
  data_size_t* left_leaf_start,
  data_size_t* right_leaf_start,
  double* left_leaf_sum_of_hessians,
  double* right_leaf_sum_of_hessians) {
  LaunchSplitInnerKernel(
    num_data_in_leaf,
    best_split_info,
    left_leaf_index,
    right_leaf_index,
    smaller_leaf_splits,
    larger_leaf_splits,
    left_leaf_num_data,
    right_leaf_num_data,
    left_leaf_start,
    right_leaf_start,
    left_leaf_sum_of_hessians,
    right_leaf_sum_of_hessians);
  ++cur_num_leaves_;
}

void CUDADataPartition::UpdateTrainScore(const double* leaf_value, double* cuda_scores) {
  LaunchAddPredictionToScoreKernel(leaf_value, cuda_scores);
}

void CUDADataPartition::CalcBlockDim(const data_size_t num_data_in_leaf) {
  const int min_num_blocks = num_data_in_leaf <= 100 ? 1 : 80;
  const int num_blocks = std::max(min_num_blocks, (num_data_in_leaf + SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION - 1) / SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION);
  int split_indices_block_size_data_partition = (num_data_in_leaf + num_blocks - 1) / num_blocks - 1;
  CHECK_GT(split_indices_block_size_data_partition, 0);
  int split_indices_block_size_data_partition_aligned = 1;
  while (split_indices_block_size_data_partition > 0) {
    split_indices_block_size_data_partition_aligned <<= 1;
    split_indices_block_size_data_partition >>= 1;
  }
  const int num_blocks_final = (num_data_in_leaf + split_indices_block_size_data_partition_aligned - 1) / split_indices_block_size_data_partition_aligned;
  grid_dim_ = num_blocks_final;
  block_dim_ = split_indices_block_size_data_partition_aligned;
}

void CUDADataPartition::SetUsedDataIndices(const data_size_t* used_indices, const data_size_t num_used_indices) {
  use_bagging_ = true;
  num_used_indices_ = num_used_indices;
  CopyFromHostToCUDADeviceOuter<data_size_t>(cuda_data_indices_, used_indices, static_cast<size_t>(num_used_indices), __FILE__, __LINE__);
}

void CUDADataPartition::SetUseBagging(const bool use_bagging) {
  use_bagging_ = use_bagging;
}

}  // namespace LightGBM

#endif  // USE_CUDA
