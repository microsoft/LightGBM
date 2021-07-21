/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_CUDA_HISTOGRAM_CONSTRUCTOR_HPP_
#define LIGHTGBM_CUDA_HISTOGRAM_CONSTRUCTOR_HPP_

#ifdef USE_CUDA

#include <LightGBM/feature_group.h>
#include <LightGBM/tree.h>

#include <fstream>

#include "new_cuda_utils.hpp"
#include "cuda_leaf_splits.hpp"

#include <vector>

#define SHRAE_HIST_SIZE (6144 * 2)
#define NUM_DATA_PER_THREAD (400)
#define NUM_THRADS_PER_BLOCK (504)
#define NUM_FEATURE_PER_THREAD_GROUP (28)
#define SUBTRACT_BLOCK_SIZE (1024)
#define FIX_HISTOGRAM_SHARED_MEM_SIZE (1024)
#define FIX_HISTOGRAM_BLOCK_SIZE (512)
#define USED_HISTOGRAM_BUFFER_NUM (8)

namespace LightGBM {

class CUDAHistogramConstructor {
 public:
  CUDAHistogramConstructor(const Dataset* train_data, const int num_leaves, const int num_threads,
    const std::vector<uint32_t>& feature_hist_offsets,
    const int min_data_in_leaf, const double min_sum_hessian_in_leaf);

  void Init(const Dataset* train_data, TrainingShareStates* share_state);

  void ConstructHistogramForLeaf(
    const CUDALeafSplitsStruct* cuda_smaller_leaf_splits, const CUDALeafSplitsStruct* cuda_larger_leaf_splits, 
    const data_size_t* cuda_leaf_num_data, const data_size_t num_data_in_smaller_leaf, const data_size_t num_data_in_larger_leaf,
    const double sum_hessians_in_smaller_leaf, const double sum_hessians_in_larger_leaf);

  void BeforeTrain(const score_t* gradients, const score_t* hessians);

  const hist_t* cuda_hist() const { return cuda_hist_; }

  hist_t* cuda_hist_pointer() const { return cuda_hist_; }

  hist_t* cuda_hist_pointer() { return cuda_hist_; }

  const uint8_t* cuda_data() const { return cuda_data_uint8_t_; }

 private:

  void CalcConstructHistogramKernelDim(int* grid_dim_x, int* grid_dim_y, int* block_dim_x, int* block_dim_y,
    const data_size_t num_data_in_smaller_leaf);

  void LaunchConstructHistogramKernel(
    const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
    const data_size_t* cuda_leaf_num_data,
    const data_size_t num_data_in_smaller_leaf);

  void LaunchSubtractHistogramKernel(
    const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
    const CUDALeafSplitsStruct* cuda_larger_leaf_splits);

  void InitCUDAData(TrainingShareStates* share_state);

  void PushOneData(const uint32_t feature_bin_value, const int feature_group_id, const data_size_t data_index);

  void DivideCUDAFeatureGroups(const Dataset* train_data, TrainingShareStates* share_state);

  template <typename BIN_TYPE>
  void GetDenseDataPartitioned(const BIN_TYPE* row_wise_data, std::vector<BIN_TYPE>* partitioned_data);

  template <typename BIN_TYPE, typename DATA_PTR_TYPE>
  void GetSparseDataPartitioned(const BIN_TYPE* row_wise_data,
    const DATA_PTR_TYPE* row_ptr,
    std::vector<std::vector<BIN_TYPE>>* partitioned_data,
    std::vector<std::vector<DATA_PTR_TYPE>>* partitioned_row_ptr,
    std::vector<DATA_PTR_TYPE>* partition_ptr);

  // Host memory
  // data on CPU, stored in row-wise style
  const data_size_t num_data_;
  const int num_features_;
  const int num_leaves_;
  const int num_threads_;
  int num_total_bin_;
  int num_feature_groups_;
  std::vector<uint8_t> data_;
  std::vector<uint32_t> feature_group_bin_offsets_;
  std::vector<uint8_t> feature_mfb_offsets_;
  std::vector<uint32_t> feature_num_bins_;
  std::vector<uint32_t> feature_hist_offsets_;
  std::vector<uint32_t> feature_most_freq_bins_;
  const int min_data_in_leaf_;
  const double min_sum_hessian_in_leaf_;
  std::vector<int> feature_partition_column_index_offsets_;
  std::vector<uint32_t> column_hist_offsets_;
  std::vector<uint32_t> column_hist_offsets_full_;
  bool is_sparse_;
  int num_feature_partitions_;
  int max_num_column_per_partition_;
  uint8_t data_ptr_bit_type_;
  uint8_t bit_type_;
  const Dataset* train_data_;
  std::vector<cudaStream_t> cuda_streams_;
  std::vector<int> need_fix_histogram_features_;
  std::vector<uint32_t> need_fix_histogram_features_num_bin_aligend_;

  const int min_grid_dim_y_ = 160;

  // CUDA memory, held by this object
  uint32_t* cuda_feature_group_bin_offsets_;
  uint8_t* cuda_feature_mfb_offsets_;
  uint32_t* cuda_feature_num_bins_;
  uint32_t* cuda_feature_hist_offsets_;
  uint32_t* cuda_feature_most_freq_bins_;
  hist_t* cuda_hist_;
  hist_t* block_cuda_hist_buffer_;
  int* cuda_num_total_bin_;
  int* cuda_num_feature_groups_;
  uint8_t* cuda_data_uint8_t_;
  uint16_t* cuda_data_uint16_t_;
  uint32_t* cuda_data_uint32_t_;
  uint16_t* cuda_row_ptr_uint16_t_;
  uint32_t* cuda_row_ptr_uint32_t_;
  uint64_t* cuda_row_ptr_uint64_t_;
  uint16_t* cuda_partition_ptr_uint16_t_;
  uint32_t* cuda_partition_ptr_uint32_t_;
  uint64_t* cuda_partition_ptr_uint64_t_;
  int* cuda_num_features_;
  score_t* cuda_ordered_gradients_;
  score_t* cuda_ordered_hessians_;
  int* cuda_feature_partition_column_index_offsets_;
  uint32_t* cuda_column_hist_offsets_;
  uint32_t* cuda_column_hist_offsets_full_;
  int* cuda_need_fix_histogram_features_;
  uint32_t* cuda_need_fix_histogram_features_num_bin_aligned_;

  // CUDA memory, held by other objects
  const score_t* cuda_gradients_;
  const score_t* cuda_hessians_;
};

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_CUDA_HISTOGRAM_CONSTRUCTOR_HPP_
