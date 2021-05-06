/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_CUDA_DATA_SPLITTER_HPP_
#define LIGHTGBM_CUDA_DATA_SPLITTER_HPP_

#ifdef USE_CUDA

#include <LightGBM/meta.h>
#include <LightGBM/tree.h>
#include "new_cuda_utils.hpp"

#define FILL_INDICES_BLOCK_SIZE_DATA_PARTITION (1024)
#define SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION (1024)
#define NUM_BANKS_DATA_PARTITION (32)
#define LOG_NUM_BANKS_DATA_PARTITION (5)

namespace LightGBM {

class CUDADataPartition {
 public:
  CUDADataPartition(const data_size_t num_data, const int num_features, const int num_leaves,
  const int num_threads, const data_size_t* cuda_num_data, const int* cuda_num_leaves, const uint8_t* cuda_data,
  const int* cuda_num_features, const std::vector<uint32_t>& feature_hist_offsets, const Dataset* train_data);

  void Init();

  void BeforeTrain(const data_size_t* data_indices);

  void Split(const int* leaf_id, const int* best_split_feature,
    const uint32_t* best_split_threshold, const uint8_t* best_split_default_left);

  Tree* GetCPUTree();

  void Test() {
    PrintLastCUDAError();
    std::vector<data_size_t> test_data_indices(num_data_, -1);
    CopyFromCUDADeviceToHost<data_size_t>(test_data_indices.data(), cuda_data_indices_, static_cast<size_t>(num_data_));
    for (data_size_t i = 0; i < num_data_; ++i) {
      CHECK_EQ(i, test_data_indices[i]);
    }
    data_size_t test_leaf_data_start_0 = 0, test_leaf_data_end_0 = 0, test_leaf_num_data_0 = 0;
    data_size_t test_leaf_data_start_1 = 0, test_leaf_data_end_1 = 0, test_leaf_num_data_1 = 0;
    CopyFromCUDADeviceToHost<data_size_t>(&test_leaf_data_start_0, cuda_leaf_data_start_, 1);
    CopyFromCUDADeviceToHost<data_size_t>(&test_leaf_data_end_0, cuda_leaf_data_end_, 1);
    CopyFromCUDADeviceToHost<data_size_t>(&test_leaf_num_data_0, cuda_leaf_num_data_, 1);
    CopyFromCUDADeviceToHost<data_size_t>(&test_leaf_data_start_1, cuda_leaf_data_start_ + 1, 1);
    CopyFromCUDADeviceToHost<data_size_t>(&test_leaf_data_end_1, cuda_leaf_data_end_ + 1, 1);
    CopyFromCUDADeviceToHost<data_size_t>(&test_leaf_num_data_1, cuda_leaf_num_data_ + 1, 1);
    Log::Warning("test_leaf_data_start_0 = %d", test_leaf_data_start_0);
    Log::Warning("test_leaf_data_end_0 = %d", test_leaf_data_end_0);
    Log::Warning("test_leaf_num_data_0 = %d", test_leaf_num_data_0);
    Log::Warning("test_leaf_data_start_1 = %d", test_leaf_data_start_1);
    Log::Warning("test_leaf_data_end_1 = %d", test_leaf_data_end_1);
    Log::Warning("test_leaf_num_data_1 = %d", test_leaf_num_data_1);
    Log::Warning("CUDADataPartition::Test Pass");
  }

  void TestAfterSplit() {
    std::vector<uint8_t> test_bit_vector(num_data_, 0);
    CopyFromCUDADeviceToHost<uint8_t>(test_bit_vector.data(), cuda_data_to_left_, static_cast<size_t>(num_data_));
    data_size_t num_data_to_left = 0;
    #pragma omp parallel for schedule(static) num_threads(num_threads_) reduction(+:num_data_to_left)
    for (data_size_t data_index = 0; data_index < num_data_; ++data_index) {
      if (test_bit_vector[data_index]) {
        ++num_data_to_left;
      }
    }
    Log::Warning("CUDADataPartition::TestAfterSplit num_data_to_left = %d", num_data_to_left);
    std::vector<data_size_t> test_data_indices(num_data_, 0);
    CopyFromCUDADeviceToHost<data_size_t>(test_data_indices.data(), cuda_data_indices_, static_cast<size_t>(num_data_));
    std::vector<int> test_leaf_num_data(num_leaves_, 0), test_leaf_data_start(num_leaves_, 0), test_leaf_data_end(num_leaves_, 0);
    CopyFromCUDADeviceToHost<int>(test_leaf_num_data.data(), cuda_leaf_num_data_, static_cast<size_t>(num_leaves_));
    CopyFromCUDADeviceToHost<int>(test_leaf_data_start.data(), cuda_leaf_data_start_, static_cast<size_t>(num_leaves_));
    CopyFromCUDADeviceToHost<int>(test_leaf_data_end.data(), cuda_leaf_data_end_, static_cast<size_t>(num_leaves_));
    for (int i = 0; i < num_leaves_; ++i) {
      Log::Warning("test_leaf_num_data[%d] = %d", i, test_leaf_num_data[i]);
      Log::Warning("test_leaf_data_start[%d] = %d", i, test_leaf_data_start[i]);
      Log::Warning("test_leaf_data_end[%d] = %d", i, test_leaf_data_end[i]);
    }
    const data_size_t num_data_in_leaf_0 = test_leaf_num_data[0];
    const int check_window_size = 10;
    for (data_size_t i = 0; i < check_window_size; ++i) {
      Log::Warning("test_data_indices[%d] = %d", i, test_data_indices[i]);
    }
    for (data_size_t i = num_data_in_leaf_0 - check_window_size; i < num_data_in_leaf_0; ++i) {
      Log::Warning("test_data_indices[%d] = %d", i, test_data_indices[i]);
    }
    for (data_size_t i = num_data_in_leaf_0; i < num_data_in_leaf_0 + check_window_size; ++i) {
      Log::Warning("test_data_indices[%d] = %d", i, test_data_indices[i]);
    }
  }

  void TestPrefixSum() {
    std::vector<uint32_t> test_elements(SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION, 1);
    uint32_t* cuda_elements = nullptr;
    InitCUDAMemoryFromHostMemory<uint32_t>(&cuda_elements, test_elements.data(), test_elements.size());
    LaunchPrefixSumKernel(cuda_elements);
    CopyFromCUDADeviceToHost<uint32_t>(test_elements.data(), cuda_elements, test_elements.size());
    for (int i = 0; i < SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION; ++i) {
      Log::Warning("test_elements[%d] = %d", i, test_elements[i]);
    }
  }

  const data_size_t* cuda_leaf_data_start() const { return cuda_leaf_data_start_; }

  const data_size_t* cuda_leaf_data_end() const { return cuda_leaf_data_end_; }

  const data_size_t* cuda_leaf_num_data() const { return cuda_leaf_num_data_; }

  //const data_size_t* cuda_leaf_num_data_offsets() const { return cuda_leaf_num_data_offsets_; }

  const data_size_t* cuda_data_indices() const { return cuda_data_indices_; }

  const int* cuda_cur_num_leaves() const { return cuda_cur_num_leaves_; }

 private:
  void GenDataToLeftBitVector(const int* leaf_id, const int* best_split_feature,
    const uint32_t* best_split_threshold, const uint8_t* best_split_default_left);

  void SplitInner(const int* leaf_index);

  // kernel launch functions
  void LaunchFillDataIndicesBeforeTrain();

  void LaunchSplitInnerKernel(const int* leaf_index);

  void LaunchGenDataToLeftBitVectorKernel(const int* leaf_index, const int* best_split_feature,
    const uint32_t* best_split_threshold, const uint8_t* best_split_default_left);

  void LaunchPrefixSumKernel(uint32_t* cuda_elements);

  // Host memory
  const data_size_t num_data_;
  const int num_features_;
  const int num_leaves_;
  const int num_threads_;
  int max_num_split_indices_blocks_;
  std::vector<uint32_t> feature_default_bins_;
  std::vector<uint32_t> feature_most_freq_bins_;
  std::vector<uint32_t> feature_max_bins_;
  std::vector<uint32_t> feature_min_bins_;
  std::vector<uint8_t> feature_missing_is_zero_;
  std::vector<uint8_t> feature_missing_is_na_;
  std::vector<uint8_t> feature_mfb_is_zero_;
  std::vector<uint8_t> feature_mfb_is_na_;

  // CUDA memory, held by this object
  data_size_t* cuda_data_indices_;
  data_size_t* cuda_leaf_data_start_;
  data_size_t* cuda_leaf_data_end_;
  data_size_t* cuda_leaf_num_data_;
  int* cuda_cur_num_leaves_;
  // for split
  uint8_t* cuda_data_to_left_;
  data_size_t* cuda_block_data_to_left_offset_;
  data_size_t* cuda_block_data_to_right_offset_;
  data_size_t* cuda_out_data_indices_in_leaf_;
  uint32_t* cuda_feature_default_bins_;
  uint32_t* cuda_feature_most_freq_bins_;
  uint32_t* cuda_feature_max_bins_;
  uint32_t* cuda_feature_min_bins_;
  uint8_t* cuda_feature_missing_is_zero_;
  uint8_t* cuda_feature_missing_is_na_;
  uint8_t* cuda_feature_mfb_is_zero_;
  uint8_t* cuda_feature_mfb_is_na_;

  // CUDA memory, held by other object
  const data_size_t* cuda_num_data_;
  const int* cuda_num_leaves_;
  const uint8_t* cuda_data_;
  const int* cuda_num_features_;
};

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_CUDA_DATA_SPLITTER_HPP_
