/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_CUDA_LEAF_SPLITS_HPP_
#define LIGHTGBM_CUDA_LEAF_SPLITS_HPP_

#ifdef USE_CUDA

#include <LightGBM/utils/log.h>
#include <LightGBM/meta.h>
#include <LightGBM/bin.h>
#include "new_cuda_utils.hpp"

#define INIT_SUM_BLOCK_SIZE_LEAF_SPLITS (6144)
#define NUM_THRADS_PER_BLOCK_LEAF_SPLITS (1024)
#define NUM_DATA_THREAD_ADD_LEAF_SPLITS (6)

namespace LightGBM {

class CUDALeafSplits {
 public:
  CUDALeafSplits(const data_size_t num_data, const int leaf_index,
    const score_t* cuda_gradients, const score_t* cuda_hessians,
    const int* cuda_num_data);

  CUDALeafSplits();

  void Init();

  void InitValues(const double* cuda_sum_of_gradients, const double* cuda_sum_of_hessians,
    const data_size_t* cuda_num_data_in_leaf, const data_size_t* cuda_data_indices_in_leaf,
    hist_t* cuda_hist_in_leaf, const double* cuda_gain, const double* cuda_leaf_value);

  void InitValues(const data_size_t* cuda_data_indices_in_leaf, hist_t* cuda_hist_in_leaf);

  void InitValues();

  const int* cuda_leaf_index() const { return cuda_leaf_index_; }

  const data_size_t** cuda_data_indices_in_leaf() const { return cuda_data_indices_in_leaf_; }

  const double* cuda_gain() const { return cuda_gain_; }

  const double* cuda_sum_of_gradients() const { return cuda_sum_of_gradients_; }

  const double* cuda_sum_of_hessians() const { return cuda_sum_of_hessians_; }

  const data_size_t* cuda_num_data_in_leaf() const { return cuda_num_data_in_leaf_; }

  int* cuda_leaf_index_pointer() const { return cuda_leaf_index_; }

  double* cuda_sum_of_gradients_pointer() const { return cuda_sum_of_gradients_; }

  double* cuda_sum_of_hessians_pointer() const { return cuda_sum_of_hessians_; }

  data_size_t* cuda_num_data_in_leaf_pointer() const { return cuda_num_data_in_leaf_; }

  double* cuda_gain_pointer() const { return cuda_gain_; }

  double* cuda_leaf_value_pointer() const { return cuda_leaf_value_; }

  const data_size_t** cuda_data_indices_in_leaf_pointer_pointer() { return cuda_data_indices_in_leaf_; }

  hist_t** cuda_hist_in_leaf_pointer_pointer() const { return cuda_hist_in_leaf_; }

  void Test() {
    PrintLastCUDAError();
    double test_sum_of_gradients = 0.0f, test_sum_of_hessians = 0.0f;
    CopyFromCUDADeviceToHost<double>(&test_sum_of_gradients, cuda_sum_of_gradients_, 1);
    CopyFromCUDADeviceToHost<double>(&test_sum_of_hessians, cuda_sum_of_hessians_, 1);
    Log::Warning("CUDALeafSplits::Test test_sum_of_gradients = %f", test_sum_of_gradients);
    Log::Warning("CUDALeafSplits::Test test_sum_of_hessians = %f", test_sum_of_hessians);
  }

 private:
  void LaunchInitValuesKernal();

  // Host memory
  const int num_data_;
  const int leaf_index_;
  int num_blocks_init_from_gradients_;

  // CUDA memory, held by this object
  int* cuda_leaf_index_;
  double* cuda_sum_of_gradients_;
  double* cuda_sum_of_hessians_;
  data_size_t* cuda_num_data_in_leaf_;
  double* cuda_gain_;
  double* cuda_leaf_value_;

  // CUDA memory, held by other object
  const data_size_t** cuda_data_indices_in_leaf_;
  hist_t** cuda_hist_in_leaf_;
  const score_t* cuda_gradients_;
  const score_t* cuda_hessians_;
  const int* cuda_num_data_;
};

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_CUDA_LEAF_SPLITS_HPP_
