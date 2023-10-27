/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include <algorithm>
#include <memory>

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
  cuda_column_data_ = train_data->cuda_column_data();

  is_categorical_feature_.resize(train_data->num_features(), false);
  is_single_feature_in_column_.resize(train_data->num_features(), false);
  for (int feature_index = 0; feature_index < train_data->num_features(); ++feature_index) {
    if (train_data->FeatureBinMapper(feature_index)->bin_type() == BinType::CategoricalBin) {
      is_categorical_feature_[feature_index] = true;
    }
    const int feature_group_index = train_data->Feature2Group(feature_index);
    if (!train_data->IsMultiGroup(feature_group_index)) {
      if ((feature_index == 0 || train_data->Feature2Group(feature_index - 1) != feature_group_index) &&
        (feature_index == train_data->num_features() - 1 || train_data->Feature2Group(feature_index + 1) != feature_group_index)) {
        is_single_feature_in_column_[feature_index] = true;
      }
    } else {
      is_single_feature_in_column_[feature_index] = true;
    }
  }

  cuda_data_indices_ = nullptr;
  cuda_leaf_data_start_ = nullptr;
  cuda_leaf_data_end_ = nullptr;
  cuda_leaf_num_data_ = nullptr;
  cuda_hist_pool_ = nullptr;
  cuda_leaf_output_ = nullptr;
  cuda_block_to_left_offset_ = nullptr;
  cuda_data_index_to_leaf_index_ = nullptr;
  cuda_block_data_to_left_offset_ = nullptr;
  cuda_block_data_to_right_offset_ = nullptr;
  cuda_out_data_indices_in_leaf_ = nullptr;
  cuda_split_info_buffer_ = nullptr;
  cuda_num_data_ = nullptr;
}

CUDADataPartition::~CUDADataPartition() {
  DeallocateCUDAMemory<data_size_t>(&cuda_data_indices_, __FILE__, __LINE__);
  DeallocateCUDAMemory<data_size_t>(&cuda_leaf_data_start_, __FILE__, __LINE__);
  DeallocateCUDAMemory<data_size_t>(&cuda_leaf_data_end_, __FILE__, __LINE__);
  DeallocateCUDAMemory<data_size_t>(&cuda_leaf_num_data_, __FILE__, __LINE__);
  DeallocateCUDAMemory<hist_t*>(&cuda_hist_pool_, __FILE__, __LINE__);
  DeallocateCUDAMemory<double>(&cuda_leaf_output_, __FILE__, __LINE__);
  DeallocateCUDAMemory<uint16_t>(&cuda_block_to_left_offset_, __FILE__, __LINE__);
  DeallocateCUDAMemory<data_size_t>(&cuda_data_index_to_leaf_index_, __FILE__, __LINE__);
  DeallocateCUDAMemory<data_size_t>(&cuda_block_data_to_left_offset_, __FILE__, __LINE__);
  DeallocateCUDAMemory<data_size_t>(&cuda_block_data_to_right_offset_, __FILE__, __LINE__);
  DeallocateCUDAMemory<data_size_t>(&cuda_out_data_indices_in_leaf_, __FILE__, __LINE__);
  DeallocateCUDAMemory<int>(&cuda_split_info_buffer_, __FILE__, __LINE__);
  DeallocateCUDAMemory<data_size_t>(&cuda_num_data_, __FILE__, __LINE__);
  CUDASUCCESS_OR_FATAL(cudaStreamDestroy(cuda_streams_[0]));
  CUDASUCCESS_OR_FATAL(cudaStreamDestroy(cuda_streams_[1]));
  CUDASUCCESS_OR_FATAL(cudaStreamDestroy(cuda_streams_[2]));
  CUDASUCCESS_OR_FATAL(cudaStreamDestroy(cuda_streams_[3]));
  cuda_streams_.clear();
  cuda_streams_.shrink_to_fit();
}

void CUDADataPartition::Init() {
  // allocate CUDA memory
  AllocateCUDAMemory<data_size_t>(&cuda_data_indices_, static_cast<size_t>(num_data_), __FILE__, __LINE__);
  AllocateCUDAMemory<data_size_t>(&cuda_leaf_data_start_, static_cast<size_t>(num_leaves_), __FILE__, __LINE__);
  AllocateCUDAMemory<data_size_t>(&cuda_leaf_data_end_, static_cast<size_t>(num_leaves_), __FILE__, __LINE__);
  AllocateCUDAMemory<data_size_t>(&cuda_leaf_num_data_, static_cast<size_t>(num_leaves_), __FILE__, __LINE__);
  // leave some space for alignment
  AllocateCUDAMemory<uint16_t>(&cuda_block_to_left_offset_, static_cast<size_t>(num_data_), __FILE__, __LINE__);
  AllocateCUDAMemory<int>(&cuda_data_index_to_leaf_index_, static_cast<size_t>(num_data_), __FILE__, __LINE__);
  AllocateCUDAMemory<data_size_t>(&cuda_block_data_to_left_offset_, static_cast<size_t>(max_num_split_indices_blocks_) + 1, __FILE__, __LINE__);
  AllocateCUDAMemory<data_size_t>(&cuda_block_data_to_right_offset_, static_cast<size_t>(max_num_split_indices_blocks_) + 1, __FILE__, __LINE__);
  SetCUDAMemory<data_size_t>(cuda_block_data_to_left_offset_, 0, static_cast<size_t>(max_num_split_indices_blocks_) + 1, __FILE__, __LINE__);
  SetCUDAMemory<data_size_t>(cuda_block_data_to_right_offset_, 0, static_cast<size_t>(max_num_split_indices_blocks_) + 1, __FILE__, __LINE__);
  AllocateCUDAMemory<data_size_t>(&cuda_out_data_indices_in_leaf_, static_cast<size_t>(num_data_), __FILE__, __LINE__);
  AllocateCUDAMemory<hist_t*>(&cuda_hist_pool_, static_cast<size_t>(num_leaves_), __FILE__, __LINE__);
  CopyFromHostToCUDADevice<hist_t*>(cuda_hist_pool_, &cuda_hist_, 1, __FILE__, __LINE__);

  AllocateCUDAMemory<int>(&cuda_split_info_buffer_, 16, __FILE__, __LINE__);

  AllocateCUDAMemory<double>(&cuda_leaf_output_, static_cast<size_t>(num_leaves_), __FILE__, __LINE__);

  cuda_streams_.resize(4);
  gpuAssert(cudaStreamCreate(&cuda_streams_[0]), __FILE__, __LINE__);
  gpuAssert(cudaStreamCreate(&cuda_streams_[1]), __FILE__, __LINE__);
  gpuAssert(cudaStreamCreate(&cuda_streams_[2]), __FILE__, __LINE__);
  gpuAssert(cudaStreamCreate(&cuda_streams_[3]), __FILE__, __LINE__);

  InitCUDAMemoryFromHostMemory<data_size_t>(&cuda_num_data_, &num_data_, 1, __FILE__, __LINE__);
  use_bagging_ = false;
  used_indices_ = nullptr;
}

void CUDADataPartition::BeforeTrain() {
  if (!use_bagging_) {
    LaunchFillDataIndicesBeforeTrain();
  }
  SetCUDAMemory<data_size_t>(cuda_leaf_num_data_, 0, static_cast<size_t>(num_leaves_), __FILE__, __LINE__);
  SetCUDAMemory<data_size_t>(cuda_leaf_data_start_, 0, static_cast<size_t>(num_leaves_), __FILE__, __LINE__);
  SetCUDAMemory<data_size_t>(cuda_leaf_data_end_, 0, static_cast<size_t>(num_leaves_), __FILE__, __LINE__);
  SynchronizeCUDADevice(__FILE__, __LINE__);
  if (!use_bagging_) {
    CopyFromCUDADeviceToCUDADevice<data_size_t>(cuda_leaf_num_data_, cuda_num_data_, 1, __FILE__, __LINE__);
    CopyFromCUDADeviceToCUDADevice<data_size_t>(cuda_leaf_data_end_, cuda_num_data_, 1, __FILE__, __LINE__);
  } else {
    CopyFromHostToCUDADevice<data_size_t>(cuda_leaf_num_data_, &num_used_indices_, 1, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<data_size_t>(cuda_leaf_data_end_, &num_used_indices_, 1, __FILE__, __LINE__);
  }
  SynchronizeCUDADevice(__FILE__, __LINE__);
  CopyFromHostToCUDADevice<hist_t*>(cuda_hist_pool_, &cuda_hist_, 1, __FILE__, __LINE__);
}

void CUDADataPartition::Split(
  // input best split info
  const CUDASplitInfo* best_split_info,
  const int left_leaf_index,
  const int right_leaf_index,
  const int leaf_best_split_feature,
  const uint32_t leaf_best_split_threshold,
  const uint32_t* categorical_bitset,
  const int categorical_bitset_len,
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
  double* right_leaf_sum_of_hessians,
  double* left_leaf_sum_of_gradients,
  double* right_leaf_sum_of_gradients) {
  CalcBlockDim(num_data_in_leaf);
  global_timer.Start("GenDataToLeftBitVector");
  GenDataToLeftBitVector(num_data_in_leaf,
                         leaf_best_split_feature,
                         leaf_best_split_threshold,
                         categorical_bitset,
                         categorical_bitset_len,
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
             right_leaf_sum_of_hessians,
             left_leaf_sum_of_gradients,
             right_leaf_sum_of_gradients);
  global_timer.Stop("SplitInner");
}

void CUDADataPartition::GenDataToLeftBitVector(
    const data_size_t num_data_in_leaf,
    const int split_feature_index,
    const uint32_t split_threshold,
    const uint32_t* categorical_bitset,
    const int categorical_bitset_len,
    const uint8_t split_default_left,
    const data_size_t leaf_data_start,
    const int left_leaf_index,
    const int right_leaf_index) {
  if (is_categorical_feature_[split_feature_index]) {
    LaunchGenDataToLeftBitVectorCategoricalKernel(
      num_data_in_leaf,
      split_feature_index,
      categorical_bitset,
      categorical_bitset_len,
      split_default_left,
      leaf_data_start,
      left_leaf_index,
      right_leaf_index);
  } else {
    LaunchGenDataToLeftBitVectorKernel(
      num_data_in_leaf,
      split_feature_index,
      split_threshold,
      split_default_left,
      leaf_data_start,
      left_leaf_index,
      right_leaf_index);
  }
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
  double* right_leaf_sum_of_hessians,
  double* left_leaf_sum_of_gradients,
  double* right_leaf_sum_of_gradients) {
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
    right_leaf_sum_of_hessians,
    left_leaf_sum_of_gradients,
    right_leaf_sum_of_gradients);
  ++cur_num_leaves_;
}

void CUDADataPartition::UpdateTrainScore(const Tree* tree, double* scores) {
  const CUDATree* cuda_tree = nullptr;
  std::unique_ptr<CUDATree> cuda_tree_ptr;
  if (tree->is_cuda_tree()) {
    cuda_tree = reinterpret_cast<const CUDATree*>(tree);
  } else {
    cuda_tree_ptr.reset(new CUDATree(tree));
    cuda_tree = cuda_tree_ptr.get();
  }
  if (use_bagging_) {
    // we need restore the order of indices in cuda_data_indices_
    CopyFromCUDADeviceToCUDADevice<data_size_t>(cuda_data_indices_, used_indices_, static_cast<size_t>(num_used_indices_), __FILE__, __LINE__);
  }
  LaunchAddPredictionToScoreKernel(cuda_tree->cuda_leaf_value(), scores);
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
  used_indices_ = used_indices;
  CopyFromCUDADeviceToCUDADevice<data_size_t>(cuda_data_indices_, used_indices, static_cast<size_t>(num_used_indices), __FILE__, __LINE__);
  LaunchFillDataIndexToLeafIndex();
}

void CUDADataPartition::ResetTrainingData(const Dataset* train_data, const int num_total_bin, hist_t* cuda_hist) {
  const data_size_t old_num_data = num_data_;
  num_data_ = train_data->num_data();
  num_features_ = train_data->num_features();
  num_total_bin_ = num_total_bin;
  cuda_column_data_ = train_data->cuda_column_data();
  cuda_hist_ = cuda_hist;
  CopyFromHostToCUDADevice<hist_t*>(cuda_hist_pool_, &cuda_hist_, 1, __FILE__, __LINE__);
  CopyFromHostToCUDADevice<int>(cuda_num_data_, &num_data_, 1, __FILE__, __LINE__);
  if (num_data_ > old_num_data) {
    CalcBlockDim(num_data_);
    const int old_max_num_split_indices_blocks = max_num_split_indices_blocks_;
    max_num_split_indices_blocks_ = grid_dim_;
    if (max_num_split_indices_blocks_ > old_max_num_split_indices_blocks) {
      DeallocateCUDAMemory<data_size_t>(&cuda_block_data_to_left_offset_, __FILE__, __LINE__);
      DeallocateCUDAMemory<data_size_t>(&cuda_block_data_to_right_offset_, __FILE__, __LINE__);
      AllocateCUDAMemory<data_size_t>(&cuda_block_data_to_left_offset_, static_cast<size_t>(max_num_split_indices_blocks_) + 1, __FILE__, __LINE__);
      AllocateCUDAMemory<data_size_t>(&cuda_block_data_to_right_offset_, static_cast<size_t>(max_num_split_indices_blocks_) + 1, __FILE__, __LINE__);
      SetCUDAMemory<data_size_t>(cuda_block_data_to_left_offset_, 0, static_cast<size_t>(max_num_split_indices_blocks_) + 1, __FILE__, __LINE__);
      SetCUDAMemory<data_size_t>(cuda_block_data_to_right_offset_, 0, static_cast<size_t>(max_num_split_indices_blocks_) + 1, __FILE__, __LINE__);
    }
    DeallocateCUDAMemory<data_size_t>(&cuda_data_indices_, __FILE__, __LINE__);
    DeallocateCUDAMemory<uint16_t>(&cuda_block_to_left_offset_, __FILE__, __LINE__);
    DeallocateCUDAMemory<int>(&cuda_data_index_to_leaf_index_, __FILE__, __LINE__);
    DeallocateCUDAMemory<data_size_t>(&cuda_out_data_indices_in_leaf_, __FILE__, __LINE__);

    AllocateCUDAMemory<data_size_t>(&cuda_data_indices_, static_cast<size_t>(num_data_), __FILE__, __LINE__);
    AllocateCUDAMemory<uint16_t>(&cuda_block_to_left_offset_, static_cast<size_t>(num_data_), __FILE__, __LINE__);
    AllocateCUDAMemory<int>(&cuda_data_index_to_leaf_index_, static_cast<size_t>(num_data_), __FILE__, __LINE__);
    AllocateCUDAMemory<data_size_t>(&cuda_out_data_indices_in_leaf_, static_cast<size_t>(num_data_), __FILE__, __LINE__);
  }
  used_indices_ = nullptr;
  use_bagging_ = false;
  num_used_indices_ = 0;
  cur_num_leaves_ = 1;
}

void CUDADataPartition::ResetConfig(const Config* config, hist_t* cuda_hist) {
  num_threads_ = OMP_NUM_THREADS();
  num_leaves_ = config->num_leaves;
  cuda_hist_ = cuda_hist;
  DeallocateCUDAMemory<data_size_t>(&cuda_leaf_data_start_, __FILE__, __LINE__);
  DeallocateCUDAMemory<data_size_t>(&cuda_leaf_data_end_, __FILE__, __LINE__);
  DeallocateCUDAMemory<data_size_t>(&cuda_leaf_num_data_, __FILE__, __LINE__);
  DeallocateCUDAMemory<hist_t*>(&cuda_hist_pool_, __FILE__, __LINE__);
  DeallocateCUDAMemory<double>(&cuda_leaf_output_, __FILE__, __LINE__);
  AllocateCUDAMemory<data_size_t>(&cuda_leaf_data_start_, static_cast<size_t>(num_leaves_), __FILE__, __LINE__);
  AllocateCUDAMemory<data_size_t>(&cuda_leaf_data_end_, static_cast<size_t>(num_leaves_), __FILE__, __LINE__);
  AllocateCUDAMemory<data_size_t>(&cuda_leaf_num_data_, static_cast<size_t>(num_leaves_), __FILE__, __LINE__);
  AllocateCUDAMemory<hist_t*>(&cuda_hist_pool_, static_cast<size_t>(num_leaves_), __FILE__, __LINE__);
  AllocateCUDAMemory<double>(&cuda_leaf_output_, static_cast<size_t>(num_leaves_), __FILE__, __LINE__);
}

void CUDADataPartition::SetBaggingSubset(const Dataset* subset) {
  num_used_indices_ = subset->num_data();
  used_indices_ = nullptr;
  use_bagging_ = true;
  cuda_column_data_ = subset->cuda_column_data();
}

void CUDADataPartition::ResetByLeafPred(const std::vector<int>& leaf_pred, int num_leaves) {
  if (leaf_pred.size() != static_cast<size_t>(num_data_)) {
    DeallocateCUDAMemory<int>(&cuda_data_index_to_leaf_index_, __FILE__, __LINE__);
    InitCUDAMemoryFromHostMemory<int>(&cuda_data_index_to_leaf_index_, leaf_pred.data(), leaf_pred.size(), __FILE__, __LINE__);
    num_data_ = static_cast<data_size_t>(leaf_pred.size());
  } else {
    CopyFromHostToCUDADevice<int>(cuda_data_index_to_leaf_index_, leaf_pred.data(), leaf_pred.size(), __FILE__, __LINE__);
  }
  num_leaves_ = num_leaves;
  cur_num_leaves_ = num_leaves;
}

void CUDADataPartition::ReduceLeafGradStat(
  const score_t* gradients, const score_t* hessians,
  CUDATree* tree, double* leaf_grad_stat_buffer, double* leaf_hess_state_buffer) const {
  LaunchReduceLeafGradStat(gradients, hessians, tree, leaf_grad_stat_buffer, leaf_hess_state_buffer);
}

}  // namespace LightGBM

#endif  // USE_CUDA
