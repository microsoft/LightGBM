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
  const data_size_t* cuda_num_data,
  hist_t* cuda_hist):

  num_data_(train_data->num_data()),
  num_features_(train_data->num_features()),
  num_total_bin_(num_total_bin),
  num_leaves_(num_leaves),
  num_threads_(num_threads),
  cuda_hist_(cuda_hist) {

  cuda_num_data_ = cuda_num_data;
  max_num_split_indices_blocks_ = (num_data_ + SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION - 1) /
    SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION;
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
  AllocateCUDAMemory<data_size_t>(static_cast<size_t>(num_data_), &cuda_data_indices_);
  AllocateCUDAMemory<data_size_t>(static_cast<size_t>(num_leaves_), &cuda_leaf_data_start_);
  AllocateCUDAMemory<data_size_t>(static_cast<size_t>(num_leaves_), &cuda_leaf_data_end_);
  AllocateCUDAMemory<data_size_t>(static_cast<size_t>(num_leaves_), &cuda_leaf_num_data_);
  InitCUDAValueFromConstant<int>(&cuda_cur_num_leaves_, 1);
  // leave some space for alignment
  AllocateCUDAMemory<uint8_t>(static_cast<size_t>(num_data_) + 1024 * 8, &cuda_data_to_left_);
  AllocateCUDAMemory<int>(static_cast<size_t>(num_data_), &cuda_data_index_to_leaf_index_);
  AllocateCUDAMemory<data_size_t>(static_cast<size_t>(max_num_split_indices_blocks_) + 1, &cuda_block_data_to_left_offset_);
  AllocateCUDAMemory<data_size_t>(static_cast<size_t>(max_num_split_indices_blocks_) + 1, &cuda_block_data_to_right_offset_);
  SetCUDAMemory<data_size_t>(cuda_block_data_to_left_offset_, 0, static_cast<size_t>(max_num_split_indices_blocks_) + 1);
  SetCUDAMemory<data_size_t>(cuda_block_data_to_right_offset_, 0, static_cast<size_t>(max_num_split_indices_blocks_) + 1);
  AllocateCUDAMemory<data_size_t>(static_cast<size_t>(num_data_), &cuda_out_data_indices_in_leaf_);
  AllocateCUDAMemory<hist_t*>(static_cast<size_t>(num_leaves_), &cuda_hist_pool_);
  CopyFromHostToCUDADevice<hist_t*>(cuda_hist_pool_, &cuda_hist_, 1);

  AllocateCUDAMemory<int>(12, &cuda_split_info_buffer_);

  AllocateCUDAMemory<int>(static_cast<size_t>(num_leaves_), &tree_split_leaf_index_);
  AllocateCUDAMemory<int>(static_cast<size_t>(num_leaves_), &tree_inner_feature_index_);
  AllocateCUDAMemory<uint32_t>(static_cast<size_t>(num_leaves_), &tree_threshold_);
  AllocateCUDAMemory<double>(static_cast<size_t>(num_leaves_), &tree_threshold_real_);
  AllocateCUDAMemory<double>(static_cast<size_t>(num_leaves_), &tree_left_output_);
  AllocateCUDAMemory<double>(static_cast<size_t>(num_leaves_), &tree_right_output_);
  AllocateCUDAMemory<data_size_t>(static_cast<size_t>(num_leaves_), &tree_left_count_);
  AllocateCUDAMemory<data_size_t>(static_cast<size_t>(num_leaves_), &tree_right_count_);
  AllocateCUDAMemory<double>(static_cast<size_t>(num_leaves_), &tree_left_sum_hessian_);
  AllocateCUDAMemory<double>(static_cast<size_t>(num_leaves_), &tree_right_sum_hessian_);
  AllocateCUDAMemory<double>(static_cast<size_t>(num_leaves_), &tree_gain_);
  AllocateCUDAMemory<uint8_t>(static_cast<size_t>(num_leaves_), &tree_default_left_);

  AllocateCUDAMemory<double>(static_cast<size_t>(num_leaves_), &cuda_leaf_output_);

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
  InitCUDAMemoryFromHostMemory<int>(&cuda_feature_num_bin_offsets_, feature_num_bin_offsets.data(), feature_num_bin_offsets.size());
  InitCUDAMemoryFromHostMemory<double>(&cuda_bin_upper_bounds_, flatten_bin_upper_bounds.data(), flatten_bin_upper_bounds.size());
}

void CUDADataPartition::BeforeTrain(const data_size_t* data_indices) {
  if (data_indices == nullptr) {
    // no bagging
    LaunchFillDataIndicesBeforeTrain();
    SetCUDAMemory<data_size_t>(cuda_leaf_num_data_, 0, static_cast<size_t>(num_leaves_));
    SetCUDAMemory<data_size_t>(cuda_leaf_data_start_, 0, static_cast<size_t>(num_leaves_));
    SetCUDAMemory<data_size_t>(cuda_leaf_data_end_, 0, static_cast<size_t>(num_leaves_));
    SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
    CopyFromCUDADeviceToCUDADevice<data_size_t>(cuda_leaf_num_data_, cuda_num_data_, 1);
    CopyFromCUDADeviceToCUDADevice<data_size_t>(cuda_leaf_data_end_, cuda_num_data_, 1);
    SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
    cur_num_leaves_ = 1;
    CopyFromHostToCUDADevice<int>(cuda_cur_num_leaves_, &cur_num_leaves_, 1);
    CopyFromHostToCUDADevice<hist_t*>(cuda_hist_pool_, &cuda_hist_, 1);
  } else {
    Log::Fatal("bagging is not supported by GPU");
  }
}

void CUDADataPartition::Split(const int* leaf_id,
  const int* best_split_feature,
  CUDASplitInfo* best_split_info,
  // for leaf splits information update
  CUDALeafSplitsStruct* smaller_leaf_splits,
  CUDALeafSplitsStruct* larger_leaf_splits,
  std::vector<data_size_t>* cpu_leaf_num_data,
  std::vector<data_size_t>* cpu_leaf_data_start,
  std::vector<double>* cpu_leaf_sum_hessians,
  const std::vector<int>& cpu_leaf_best_split_feature,
  const std::vector<uint32_t>& cpu_leaf_best_split_threshold,
  const std::vector<uint8_t>& cpu_leaf_best_split_default_left,
  int* smaller_leaf_index, int* larger_leaf_index,
  const int cpu_leaf_index, const int cur_max_leaf_index) {
  global_timer.Start("GenDataToLeftBitVector");
  global_timer.Start("SplitInner Copy CUDA To Host");
  const data_size_t num_data_in_leaf = cpu_leaf_num_data->at(cpu_leaf_index);
  const int split_feature_index = cpu_leaf_best_split_feature[cpu_leaf_index];
  const uint32_t split_threshold = cpu_leaf_best_split_threshold[cpu_leaf_index];
  const uint8_t split_default_left = cpu_leaf_best_split_default_left[cpu_leaf_index];
  const data_size_t leaf_data_start = cpu_leaf_data_start->at(cpu_leaf_index);
  global_timer.Stop("SplitInner Copy CUDA To Host");
  GenDataToLeftBitVector(num_data_in_leaf, split_feature_index, split_threshold, split_default_left, leaf_data_start, cpu_leaf_index, cur_max_leaf_index);
  global_timer.Stop("GenDataToLeftBitVector");
  global_timer.Start("SplitInner");

  SplitInner(leaf_id, num_data_in_leaf,
    best_split_feature,
    best_split_info,
    smaller_leaf_splits,
    larger_leaf_splits,
    cpu_leaf_num_data, cpu_leaf_data_start, cpu_leaf_sum_hessians,
    smaller_leaf_index, larger_leaf_index, cpu_leaf_index);
  global_timer.Stop("SplitInner");
}

void CUDADataPartition::GenDataToLeftBitVector(const data_size_t num_data_in_leaf,
    const int split_feature_index, const uint32_t split_threshold,
    const uint8_t split_default_left, const data_size_t leaf_data_start,
    const int left_leaf_index, const int right_leaf_index) {
  LaunchGenDataToLeftBitVectorKernel(num_data_in_leaf, split_feature_index, split_threshold, split_default_left, leaf_data_start, left_leaf_index, right_leaf_index);
}

void CUDADataPartition::SplitInner(const int* leaf_index, const data_size_t num_data_in_leaf,
  const int* best_split_feature,
  CUDASplitInfo* best_split_info,
  // for leaf splits information update
  CUDALeafSplitsStruct* smaller_leaf_splits,
  CUDALeafSplitsStruct* larger_leaf_splits,
  std::vector<data_size_t>* cpu_leaf_num_data, std::vector<data_size_t>* cpu_leaf_data_start,
  std::vector<double>* cpu_leaf_sum_hessians,
  int* smaller_leaf_index, int* larger_leaf_index, const int cpu_leaf_index) {
  LaunchSplitInnerKernel(leaf_index, num_data_in_leaf,
    best_split_feature,
    best_split_info,
    smaller_leaf_splits,
    larger_leaf_splits,
    cpu_leaf_num_data, cpu_leaf_data_start, cpu_leaf_sum_hessians,
    smaller_leaf_index, larger_leaf_index, cpu_leaf_index);
  ++cur_num_leaves_;
}

Tree* CUDADataPartition::GetCPUTree() {}

void CUDADataPartition::UpdateTrainScore(const double learning_rate, double* cuda_scores) {
  LaunchAddPredictionToScoreKernel(learning_rate, cuda_scores);
}

void CUDADataPartition::CalcBlockDim(const data_size_t num_data_in_leaf,
    int* grid_dim,
    int* block_dim) {
  const int num_threads_per_block = SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION;
  const int min_grid_dim = num_data_in_leaf <= 100 ? 1 : 10;
  const int num_data_per_block = (num_threads_per_block * 8);
  const int num_blocks = std::max(min_grid_dim, (num_data_in_leaf + num_data_per_block - 1) / num_data_per_block);
  const int num_threads_per_block_final = (num_data_in_leaf + (num_blocks * 8) - 1) / (num_blocks * 8);
  int num_threads_per_block_final_ref = num_threads_per_block_final - 1;
  CHECK_GT(num_threads_per_block_final_ref, 0);
  int num_threads_per_block_final_aligned = 1;
  while (num_threads_per_block_final_ref > 0) {
    num_threads_per_block_final_aligned <<= 1;
    num_threads_per_block_final_ref >>= 1;
  }
  const int num_blocks_final = (num_data_in_leaf + (num_threads_per_block_final_aligned * 8) - 1) / (num_threads_per_block_final_aligned * 8);
  *grid_dim = num_blocks_final;
  *block_dim = num_threads_per_block_final_aligned;
}

void CUDADataPartition::CalcBlockDimInCopy(const data_size_t num_data_in_leaf,
    int* grid_dim,
    int* block_dim) {
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
  *grid_dim = num_blocks_final;
  *block_dim = split_indices_block_size_data_partition_aligned;
}

}  // namespace LightGBM

#endif  // USE_CUDA
