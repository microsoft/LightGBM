/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#include "cuda_leaf_splits.hpp"

namespace LightGBM {

CUDALeafSplits::CUDALeafSplits(const data_size_t num_data,
  const score_t* cuda_gradients, const score_t* cuda_hessians,
  const int* cuda_num_data): num_data_(num_data) {
  cuda_sum_of_gradients_ = nullptr;
  cuda_sum_of_hessians_ = nullptr;
  cuda_num_data_in_leaf_ = nullptr;
  cuda_gain_ = nullptr;
  cuda_leaf_value_ = nullptr;

  cuda_gradients_ = cuda_gradients;
  cuda_hessians_ = cuda_hessians;
  cuda_data_indices_in_leaf_ = nullptr;
  cuda_num_data_ = cuda_num_data;
}

void CUDALeafSplits::Init() {
  num_blocks_init_from_gradients_ = (num_data_ + INIT_SUM_BLOCK_SIZE_LEAF_SPLITS - 1) / INIT_SUM_BLOCK_SIZE_LEAF_SPLITS;

  // allocate more memory for sum reduction in CUDA
  // only the first element records the final sum
  AllocateCUDAMemory<double>(num_blocks_init_from_gradients_, &cuda_sum_of_gradients_);
  AllocateCUDAMemory<double>(num_blocks_init_from_gradients_, &cuda_sum_of_hessians_);

  AllocateCUDAMemory<data_size_t>(1, &cuda_num_data_in_leaf_);
  AllocateCUDAMemory<double>(1, &cuda_gain_);
  AllocateCUDAMemory<double>(1, &cuda_leaf_value_);
}

void CUDALeafSplits::InitValues(const double* cuda_sum_of_gradients, const double* cuda_sum_of_hessians,
  const data_size_t* cuda_num_data_in_leaf, const data_size_t* cuda_data_indices_in_leaf,
  const double* cuda_gain, const double* cuda_leaf_value) {
  CopyFromCUDADeviceToCUDADevice<double>(cuda_sum_of_gradients_, cuda_sum_of_gradients, 1);
  CopyFromCUDADeviceToCUDADevice<double>(cuda_sum_of_hessians_, cuda_sum_of_hessians, 1);
  CopyFromCUDADeviceToCUDADevice<data_size_t>(cuda_num_data_in_leaf_, cuda_num_data_in_leaf, 1);
  cuda_data_indices_in_leaf_ = cuda_data_indices_in_leaf;
  CopyFromCUDADeviceToCUDADevice<double>(cuda_gain_, cuda_gain, 1);
  CopyFromCUDADeviceToCUDADevice<double>(cuda_leaf_value_, cuda_leaf_value, 1);
}

void CUDALeafSplits::InitValues() {
  
}

}  // namespace LightGBM
