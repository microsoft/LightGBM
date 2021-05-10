/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifndef LIGHTGBM_CUDA_BEST_SPLIT_FINDER_HPP_
#define LIGHTGBM_CUDA_BEST_SPLIT_FINDER_HPP_

#ifdef USE_CUDA

#include "new_cuda_utils.hpp"
#include "cuda_leaf_splits.hpp"

#include <LightGBM/bin.h>
#include <LightGBM/dataset.h>

#include <vector>

#define MAX_NUM_BIN_IN_FEATURE (256)

namespace LightGBM {

class CUDABestSplitFinder {
 public:
  CUDABestSplitFinder(const hist_t* cuda_hist, const Dataset* train_data,
    const std::vector<uint32_t>& feature_hist_offsets, const int num_leaves,
    const double lambda_l1, const double lambda_l2, const data_size_t min_data_in_leaf,
    const double min_sum_hessian_in_leaf, const double min_gain_to_split,
    const int* cuda_num_features);

  void Init();

  void BeforeTrain();

  void FindBestSplitsForLeaf(const CUDALeafSplits* smaller_leaf_splits, const CUDALeafSplits* larger_leaf_splits);

  void FindBestFromAllSplits(const int* cuda_cur_num_leaves);

  const int* cuda_best_leaf() const { return cuda_best_leaf_; }

  const int* cuda_leaf_best_split_feature() const { return cuda_leaf_best_split_feature_; }

  const uint32_t* cuda_leaf_best_split_threshold() const { return cuda_leaf_best_split_threshold_; }

  const uint8_t* cuda_leaf_best_split_default_left() const { return cuda_leaf_best_split_default_left_; }

  const double* cuda_leaf_best_split_gain() const { return cuda_leaf_best_split_gain_; }

  const double* cuda_leaf_best_split_left_sum_gradient() const { return cuda_leaf_best_split_left_sum_gradient_; }

  const double* cuda_leaf_best_split_left_sum_hessian() const { return cuda_leaf_best_split_left_sum_hessian_; }

  const data_size_t* cuda_leaf_best_split_left_count() const { return cuda_leaf_best_split_left_count_; }

  const double* cuda_leaf_best_split_left_gain() const { return cuda_leaf_best_split_left_gain_; }

  const double* cuda_leaf_best_split_left_output() const { return cuda_leaf_best_split_left_output_; }

  const double* cuda_leaf_best_split_right_sum_gradient() const { return cuda_leaf_best_split_right_sum_gradient_; }

  const double* cuda_leaf_best_split_right_sum_hessian() const { return cuda_leaf_best_split_right_sum_hessian_; }

  const data_size_t* cuda_leaf_best_split_right_count() const { return cuda_leaf_best_split_right_count_; }

  const double* cuda_leaf_best_split_right_gain() const { return cuda_leaf_best_split_right_gain_; }

  const double* cuda_leaf_best_split_right_output() const { return cuda_leaf_best_split_right_output_; }

  void TestAfterInit() {
    PrintLastCUDAError();
  }

  void TestAfterFindBestSplits() {
    PrintLastCUDAError();
    const size_t feature_best_split_info_buffer_size = static_cast<size_t>(num_features_) * 4;
    std::vector<uint32_t> test_best_split_threshold(feature_best_split_info_buffer_size, 0);
    std::vector<uint8_t> test_best_split_found(feature_best_split_info_buffer_size, 0);
    CopyFromCUDADeviceToHost<uint32_t>(test_best_split_threshold.data(),
      cuda_best_split_threshold_, feature_best_split_info_buffer_size);
    CopyFromCUDADeviceToHost<uint8_t>(test_best_split_found.data(),
      cuda_best_split_found_, feature_best_split_info_buffer_size);
    for (size_t i = 0; i < feature_best_split_info_buffer_size; ++i) {
      Log::Warning("test_best_split_threshold[%d] = %d", i, test_best_split_threshold[i]);
      Log::Warning("test_best_split_found[%d] = %d", i, test_best_split_found[i]);
    }

    int test_best_leaf = 0;
    CopyFromCUDADeviceToHost<int>(&test_best_leaf, cuda_best_leaf_, 1);
    Log::Warning("test_best_leaf = %d", test_best_leaf);
  }

 private:
  void LaunchFindBestSplitsForLeafKernel(const int* smaller_leaf_id, const int* larger_leaf_id,
    const double* smaller_leaf_gain, const double* larger_leaf_gain, const double* sum_gradients_in_smaller_leaf,
    const double* sum_hessians_in_smaller_leaf, const data_size_t* num_data_in_smaller_leaf, hist_t** smaller_leaf_hist,
    const double* sum_gradients_in_larger_leaf, const double* sum_hessians_in_larger_leaf,
    const data_size_t* num_data_in_larger_leaf, hist_t** larger_leaf_hist);

  void LaunchSyncBestSplitForLeafKernel(const int* smaller_leaf_index, const int* larger_leaf_index);

  void LaunchFindBestFromAllSplitsKernel(const int* cuda_cur_num_leaves);

  // Host memory
  const int num_features_;
  const int num_leaves_;
  const int num_total_bin_;
  int max_num_bin_in_feature_;
  std::vector<uint32_t> feature_hist_offsets_;
  std::vector<uint8_t> feature_mfb_offsets_;
  std::vector<uint32_t> feature_default_bins_;
  // None --> 0, Zero --> 1, NaN --> 2
  std::vector<uint8_t> feature_missing_type_;
  const double lambda_l1_;
  const double lambda_l2_;
  const data_size_t min_data_in_leaf_;
  const double min_sum_hessian_in_leaf_;
  const double min_gain_to_split_;

  // CUDA memory, held by this object
  // for per leaf best split information
  int* cuda_best_leaf_;
  int* cuda_leaf_best_split_feature_;
  uint8_t* cuda_leaf_best_split_default_left_;
  uint32_t* cuda_leaf_best_split_threshold_;
  double* cuda_leaf_best_split_gain_;
  double* cuda_leaf_best_split_left_sum_gradient_;
  double* cuda_leaf_best_split_left_sum_hessian_;
  data_size_t* cuda_leaf_best_split_left_count_;
  double* cuda_leaf_best_split_left_gain_;
  double* cuda_leaf_best_split_left_output_;
  double* cuda_leaf_best_split_right_sum_gradient_;
  double* cuda_leaf_best_split_right_sum_hessian_;
  data_size_t* cuda_leaf_best_split_right_count_;
  double* cuda_leaf_best_split_right_gain_;
  double* cuda_leaf_best_split_right_output_;
  // for best split information when finding best split
  int* cuda_best_split_feature_;
  uint8_t* cuda_best_split_default_left_;
  uint32_t* cuda_best_split_threshold_;
  double* cuda_best_split_gain_;
  double* cuda_best_split_left_sum_gradient_;
  double* cuda_best_split_left_sum_hessian_;
  data_size_t* cuda_best_split_left_count_;
  double* cuda_best_split_left_gain_;
  double* cuda_best_split_left_output_;
  double* cuda_best_split_right_sum_gradient_;
  double* cuda_best_split_right_sum_hessian_;
  data_size_t* cuda_best_split_right_count_;
  double* cuda_best_split_right_gain_;
  double* cuda_best_split_right_output_;
  uint8_t* cuda_best_split_found_;
  int* cuda_num_total_bin_;
  // TODO(shiyu1994): use prefix sum to accelerate best split finding
  hist_t* prefix_sum_hist_left_;
  hist_t* prefix_sum_hist_right_;
  // feature information
  uint32_t* cuda_feature_hist_offsets_;
  uint8_t* cuda_feature_mfb_offsets_;
  uint32_t* cuda_feature_default_bins_;
  uint8_t* cuda_feature_missing_type_;
  double* cuda_lambda_l1_;
  double* cuda_lambda_l2_;
  data_size_t* cuda_min_data_in_leaf_;
  double* cuda_min_sum_hessian_in_leaf_;
  double* cuda_min_gain_to_split_;

  // CUDA memory, held by other object
  const hist_t* cuda_hist_;
  const int* cuda_num_features_;
};

}

#endif  // USE_CUDA
#endif  // LIGHTGBM_CUDA_HISTOGRAM_CONSTRUCTOR_HPP_
