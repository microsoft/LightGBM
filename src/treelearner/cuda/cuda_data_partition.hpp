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
#include <LightGBM/bin.h>
#include "new_cuda_utils.hpp"
#include "cuda_leaf_splits.hpp"

#define FILL_INDICES_BLOCK_SIZE_DATA_PARTITION (1024)
#define SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION (512)
#define NUM_BANKS_DATA_PARTITION (32)
#define LOG_NUM_BANKS_DATA_PARTITION (5)

namespace LightGBM {

class CUDADataPartition {
 public:
  CUDADataPartition(const data_size_t num_data, const int num_features, const int num_leaves,
  const int num_threads, const data_size_t* cuda_num_data, const int* cuda_num_leaves,
  const int* cuda_num_features, const std::vector<uint32_t>& feature_hist_offsets, const Dataset* train_data,
  hist_t* cuda_hist);

  void Init(const Dataset* train_data);

  void BeforeTrain(const data_size_t* data_indices);

  void Split(const int* leaf_id, const double* best_split_gain, const int* best_split_feature,
    const uint32_t* best_split_threshold, const uint8_t* best_split_default_left,
    const double* best_left_sum_gradients, const double* best_left_sum_hessians, const data_size_t* best_left_count,
    const double* best_left_gain, const double* best_left_leaf_value,
    const double* best_right_sum_gradients, const double* best_right_sum_hessians, const data_size_t* best_right_count,
    const double* best_right_gain, const double* best_right_leaf_value,
    uint8_t* best_split_found,
    // for splits information update
    int* smaller_leaf_cuda_leaf_index_pointer, double* smaller_leaf_cuda_sum_of_gradients_pointer,
    double* smaller_leaf_cuda_sum_of_hessians_pointer, data_size_t* smaller_leaf_cuda_num_data_in_leaf_pointer,
    double* smaller_leaf_cuda_gain_pointer, double* smaller_leaf_cuda_leaf_value_pointer,
    const data_size_t** smaller_leaf_cuda_data_indices_in_leaf_pointer_pointer,
    hist_t** smaller_leaf_cuda_hist_pointer_pointer,
    int* larger_leaf_cuda_leaf_index_pointer, double* larger_leaf_cuda_sum_of_gradients_pointer,
    double* larger_leaf_cuda_sum_of_hessians_pointer, data_size_t* larger_leaf_cuda_num_data_in_leaf_pointer,
    double* larger_leaf_cuda_gain_pointer, double* larger_leaf_cuda_leaf_value_pointer,
    const data_size_t** larger_leaf_cuda_data_indices_in_leaf_pointer_pointer,
    hist_t** larger_leaf_cuda_hist_pointer_pointer,
    std::vector<data_size_t>* cpu_leaf_num_data,
    std::vector<data_size_t>* cpu_leaf_data_start,
    std::vector<double>* cpu_leaf_sum_hessians,
    const std::vector<int>& cpu_leaf_best_split_feature,
    const std::vector<uint32_t>& cpu_leaf_best_split_threshold,
    const std::vector<uint8_t>& cpu_leaf_best_split_default_left,
    int* smaller_leaf_index, int* larger_leaf_index,
    const int cpu_leaf_index);

  void CUDACheck(
    const int smaller_leaf_index,
    const int larger_leaf_index,
    const std::vector<data_size_t>& num_data_in_leaf,
    const CUDALeafSplits* smaller_leaf_splits,
    const CUDALeafSplits* larger_leaf_splits,
    const score_t* gradients,
    const score_t* hessians);

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
    /*for (int i = 0; i < num_leaves_; ++i) {
      Log::Warning("test_leaf_num_data[%d] = %d", i, test_leaf_num_data[i]);
      Log::Warning("test_leaf_data_start[%d] = %d", i, test_leaf_data_start[i]);
      Log::Warning("test_leaf_data_end[%d] = %d", i, test_leaf_data_end[i]);
    }*/
    const data_size_t start_pos = test_leaf_data_start[2];
    const int check_window_size = 10;
    for (data_size_t i = 0; i < check_window_size; ++i) {
      Log::Warning("test_data_indices[%d] = %d", i, test_data_indices[i]);
    }
    Log::Warning("==========================================================");
    for (data_size_t i = start_pos - check_window_size; i < start_pos; ++i) {
      Log::Warning("test_data_indices[%d] = %d", i, test_data_indices[i]);
    }
    Log::Warning("==========================================================");
    for (data_size_t i = start_pos; i < start_pos + check_window_size; ++i) {
      Log::Warning("test_data_indices[%d] = %d", i, test_data_indices[i]);
    }
    Log::Warning("==========================================================");
    const data_size_t end_pos = test_leaf_data_end[2];
    for (data_size_t i = end_pos - check_window_size; i < end_pos; ++i) {
      Log::Warning("test_data_indices[%d] = %d", i, test_data_indices[i]);
    }
    Log::Warning("==========================================================");
    for (data_size_t i = end_pos; i < end_pos + check_window_size; ++i) {
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

  void UpdateTrainScore(const double learning_rate, double* cuda_scores);

  const data_size_t* cuda_leaf_data_start() const { return cuda_leaf_data_start_; }

  const data_size_t* cuda_leaf_data_end() const { return cuda_leaf_data_end_; }

  const data_size_t* cuda_leaf_num_data() const { return cuda_leaf_num_data_; }

  //const data_size_t* cuda_leaf_num_data_offsets() const { return cuda_leaf_num_data_offsets_; }

  const data_size_t* cuda_data_indices() const { return cuda_data_indices_; }

  const int* cuda_cur_num_leaves() const { return cuda_cur_num_leaves_; }

  const int* tree_split_leaf_index() const { return tree_split_leaf_index_; }

  const int* tree_inner_feature_index() const { return tree_inner_feature_index_; }

  const uint32_t* tree_threshold() const { return tree_threshold_; }

  const double* tree_left_output() const { return tree_left_output_; }

  const double* tree_right_output() const { return tree_right_output_; }

  const data_size_t* tree_left_count() const { return tree_left_count_; }

  const data_size_t* tree_right_count() const { return tree_right_count_; }

  const double* tree_left_sum_hessian() const { return tree_left_sum_hessian_; }

  const double* tree_right_sum_hessian() const { return tree_right_sum_hessian_; }

  const double* tree_gain() const { return tree_gain_; }

  const uint8_t* tree_default_left() const { return tree_default_left_; }

 private:
  void CopyColWiseData(const Dataset* train_data);

  void GenDataToLeftBitVector(const data_size_t num_data_in_leaf,
    const int split_feature_index, const uint32_t split_threshold,
    const uint8_t split_default_left, const data_size_t leaf_data_start);

  void SplitInner(const int* leaf_index, const data_size_t num_data_in_leaf,
    const int* best_split_feature, const uint32_t* best_split_threshold,
    const uint8_t* best_split_default_left, const double* best_split_gain,
    const double* best_left_sum_gradients, const double* best_left_sum_hessians, const data_size_t* best_left_count,
    const double* best_left_gain, const double* best_left_leaf_value,
    const double* best_right_sum_gradients, const double* best_right_sum_hessians, const data_size_t* best_right_count,
    const double* best_right_gain, const double* best_right_leaf_value, uint8_t* best_split_found,
    // for leaf splits information update
    int* smaller_leaf_cuda_leaf_index_pointer, double* smaller_leaf_cuda_sum_of_gradients_pointer,
    double* smaller_leaf_cuda_sum_of_hessians_pointer, data_size_t* smaller_leaf_cuda_num_data_in_leaf_pointer,
    double* smaller_leaf_cuda_gain_pointer, double* smaller_leaf_cuda_leaf_value_pointer,
    const data_size_t** smaller_leaf_cuda_data_indices_in_leaf_pointer_pointer,
    hist_t** smaller_leaf_cuda_hist_pointer_pointer,
    int* larger_leaf_cuda_leaf_index_pointer, double* larger_leaf_cuda_sum_of_gradients_pointer,
    double* larger_leaf_cuda_sum_of_hessians_pointer, data_size_t* larger_leaf_cuda_num_data_in_leaf_pointer,
    double* larger_leaf_cuda_gain_pointer, double* larger_leaf_cuda_leaf_value_pointer,
    const data_size_t** larger_leaf_cuda_data_indices_in_leaf_pointer_pointer,
    hist_t** larger_leaf_cuda_hist_pointer_pointer,
    std::vector<data_size_t>* cpu_leaf_num_data, std::vector<data_size_t>* cpu_leaf_data_start,
    std::vector<double>* cpu_leaf_sum_hessians,
    int* smaller_leaf_index, int* larger_leaf_index);

  // kernel launch functions
  void LaunchFillDataIndicesBeforeTrain();

  void LaunchSplitInnerKernel(const int* leaf_index, const data_size_t num_data_in_leaf,
    const int* best_split_feature, const uint32_t* best_split_threshold,
    const uint8_t* best_split_default_left, const double* best_split_gain,
    const double* best_left_sum_gradients, const double* best_left_sum_hessians, const data_size_t* best_left_count,
    const double* best_left_gain, const double* best_left_leaf_value,
    const double* best_right_sum_gradients, const double* best_right_sum_hessians, const data_size_t* best_right_count,
    const double* best_right_gain, const double* best_right_leaf_value, uint8_t* best_split_found,
    // for leaf splits information update
    int* smaller_leaf_cuda_leaf_index_pointer, double* smaller_leaf_cuda_sum_of_gradients_pointer,
    double* smaller_leaf_cuda_sum_of_hessians_pointer, data_size_t* smaller_leaf_cuda_num_data_in_leaf_pointer,
    double* smaller_leaf_cuda_gain_pointer, double* smaller_leaf_cuda_leaf_value_pointer,
    const data_size_t** smaller_leaf_cuda_data_indices_in_leaf_pointer_pointer,
    hist_t** smaller_leaf_cuda_hist_pointer_pointer,
    int* larger_leaf_cuda_leaf_index_pointer, double* larger_leaf_cuda_sum_of_gradients_pointer,
    double* larger_leaf_cuda_sum_of_hessians_pointer, data_size_t* larger_leaf_cuda_num_data_in_leaf_pointer,
    double* larger_leaf_cuda_gain_pointer, double* larger_leaf_cuda_leaf_value_pointer,
    const data_size_t** larger_leaf_cuda_data_indices_in_leaf_pointer_pointer,
    hist_t** larger_leaf_cuda_hist_pointer_pointer,
    std::vector<data_size_t>* cpu_leaf_num_data, std::vector<data_size_t>* cpu_leaf_data_start,
    std::vector<double>* cpu_leaf_sum_hessians,
    int* smaller_leaf_index, int* larger_leaf_index);

  void LaunchGenDataToLeftBitVectorKernel(const data_size_t num_data_in_leaf,
    const int split_feature_index, const uint32_t split_threshold,
    const uint8_t split_default_left, const data_size_t leaf_data_start);

  void LaunchPrefixSumKernel(uint32_t* cuda_elements);

  void LaunchAddPredictionToScoreKernel(const double learning_rate, double* cuda_scores);

  void LaunchCUDACheckKernel(
    const int smaller_leaf_index,
    const int larger_leaf_index,
    const std::vector<data_size_t>& num_data_in_leaf,
    const CUDALeafSplits* smaller_leaf_splits,
    const CUDALeafSplits* larger_leaf_splits,
    const score_t* gradients,
    const score_t* hessians);

  // Host memory
  const data_size_t num_data_;
  const int num_features_;
  const int num_leaves_;
  const int num_threads_;
  const int num_total_bin_;
  int max_num_split_indices_blocks_;
  std::vector<uint32_t> feature_default_bins_;
  std::vector<uint32_t> feature_most_freq_bins_;
  std::vector<uint32_t> feature_max_bins_;
  std::vector<uint32_t> feature_min_bins_;
  std::vector<uint8_t> feature_missing_is_zero_;
  std::vector<uint8_t> feature_missing_is_na_;
  std::vector<uint8_t> feature_mfb_is_zero_;
  std::vector<uint8_t> feature_mfb_is_na_;
  std::vector<data_size_t> num_data_in_leaf_;
  int cur_num_leaves_;
  std::vector<int> cpu_split_info_buffer_;
  std::vector<uint8_t> column_bit_type_;
  std::vector<int> feature_index_to_column_index_;
  const Dataset* train_data_;

  // CUDA streams
  std::vector<cudaStream_t> cuda_streams_;

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
  int* cuda_num_total_bin_;
  int* cuda_split_info_buffer_; // prepared to be copied to cpu
  // for histogram pool
  hist_t** cuda_hist_pool_;
  // for tree structure
  int* tree_split_leaf_index_;
  int* tree_inner_feature_index_;
  uint32_t* tree_threshold_;
  double* tree_left_output_;
  double* tree_right_output_;
  data_size_t* tree_left_count_;
  data_size_t* tree_right_count_;
  double* tree_left_sum_hessian_;
  double* tree_right_sum_hessian_;
  double* tree_gain_;
  uint8_t* tree_default_left_;
  double* data_partition_leaf_output_;
  // for debug
  double* cuda_gradients_sum_buffer_;
  double* cuda_hessians_sum_buffer_;
  // for train data split
  std::vector<void*> cuda_data_by_column_;

  // CUDA memory, held by other object
  const data_size_t* cuda_num_data_;
  const int* cuda_num_leaves_;
  const int* cuda_num_features_;
  hist_t* cuda_hist_;
};

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_CUDA_DATA_SPLITTER_HPP_
