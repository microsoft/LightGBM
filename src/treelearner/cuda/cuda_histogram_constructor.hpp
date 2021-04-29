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

#include "new_cuda_utils.hpp"

#include <vector>

#define SHRAE_HIST_SIZE (6144 * 2)
#define NUM_DATA_PER_THREAD (400)
#define NUM_THRADS_PER_BLOCK (504)
#define NUM_FEATURE_PER_THREAD_GROUP (28)

namespace LightGBM {

class CUDAHistogramConstructor {
 public:
  CUDAHistogramConstructor(const Dataset* train_data, const int num_leaves, const int num_threads,
    const score_t* cuda_gradients, const score_t* cuda_hessians);

  void Init(const Dataset* train_data);

  void ConstructHistogramForLeaf(const int* cuda_smaller_leaf_index, const int* cuda_larger_leaf_index,
    const data_size_t* cuda_data_indices_in_smaller_leaf, const data_size_t* cuda_data_indices_in_larger_leaf,
    const data_size_t* cuda_leaf_num_data_offsets);

  void LaunchConstructHistogramKernel(const int* cuda_leaf_index,
    const data_size_t* cuda_data_indices_in_leaf,
    const data_size_t* cuda_leaf_num_data_offsets);

  const hist_t* cuda_hist() const { return cuda_hist_; }

  void TestAfterInit() {
    std::vector<uint8_t> test_data(data_.size(), 0);
    CopyFromCUDADeviceToHost(test_data.data(), cuda_data_, data_.size());
    for (size_t i = 0; i < 100; ++i) {
      Log::Warning("CUDAHistogramConstructor::TestAfterInit test_data[%d] = %d", i, test_data[i]);
    }
  }

 private:
  void InitCUDAData(const Dataset* train_data);

  void PushOneData(const uint32_t feature_bin_value, const int feature_group_id, const data_size_t data_index);

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

  // CUDA memory, held by this object
  uint32_t* cuda_feature_group_bin_offsets_;
  hist_t* cuda_hist_;
  int* cuda_num_total_bin_;
  int* cuda_num_feature_groups_;
  uint8_t* cuda_data_;

  // CUDA memory, held by other objects
  const score_t* cuda_gradients_;
  const score_t* cuda_hessians_;
};

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_CUDA_HISTOGRAM_CONSTRUCTOR_HPP_
