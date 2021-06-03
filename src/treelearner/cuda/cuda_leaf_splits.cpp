/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_leaf_splits.hpp"

namespace LightGBM {

CUDALeafSplits::CUDALeafSplits(const data_size_t num_data, const int leaf_index,
  const score_t* cuda_gradients, const score_t* cuda_hessians,
  const int* cuda_num_data): num_data_(num_data), leaf_index_(leaf_index) {
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

  InitCUDAMemoryFromHostMemory<data_size_t>(&cuda_num_data_in_leaf_, &num_data_, 1);
  // TODO(shiyu1994): should initialize root gain for min_gain_shift
  InitCUDAValueFromConstant<double>(&cuda_gain_, 0.0f);
  // since smooth is not used, so the output value for root node is useless
  InitCUDAValueFromConstant<double>(&cuda_leaf_value_, 0.0f);
  AllocateCUDAMemory<const data_size_t*>(1, &cuda_data_indices_in_leaf_);
  AllocateCUDAMemory<hist_t*>(1, &cuda_hist_in_leaf_);

  InitCUDAMemoryFromHostMemory<int>(&cuda_leaf_index_, &leaf_index_, 1);

  cuda_streams_.resize(2);
  CUDASUCCESS_OR_FATAL(cudaStreamCreate(&cuda_streams_[0]));
  CUDASUCCESS_OR_FATAL(cudaStreamCreate(&cuda_streams_[1]));
}

void CUDALeafSplits::InitValues(const double* cuda_sum_of_gradients, const double* cuda_sum_of_hessians,
  const data_size_t* cuda_num_data_in_leaf, const data_size_t* cuda_data_indices_in_leaf, hist_t* cuda_hist_in_leaf,
  const double* cuda_gain, const double* cuda_leaf_value) {
  CopyFromCUDADeviceToCUDADevice<double>(cuda_sum_of_gradients_, cuda_sum_of_gradients, 1);
  CopyFromCUDADeviceToCUDADevice<double>(cuda_sum_of_hessians_, cuda_sum_of_hessians, 1);
  CopyFromCUDADeviceToCUDADevice<data_size_t>(cuda_num_data_in_leaf_, cuda_num_data_in_leaf, 1);
  CopyFromHostToCUDADevice<const data_size_t*>(cuda_data_indices_in_leaf_, &cuda_data_indices_in_leaf, 1);
  CopyFromHostToCUDADevice<hist_t*>(cuda_hist_in_leaf_, &cuda_hist_in_leaf, 1);
  CopyFromCUDADeviceToCUDADevice<double>(cuda_gain_, cuda_gain, 1);
  CopyFromCUDADeviceToCUDADevice<double>(cuda_leaf_value_, cuda_leaf_value, 1);
  SynchronizeCUDADevice();
}

void CUDALeafSplits::InitValues() {
  SetCUDAMemory<double>(cuda_sum_of_gradients_, 0, num_blocks_init_from_gradients_);
  SetCUDAMemory<double>(cuda_sum_of_hessians_, 0, num_blocks_init_from_gradients_);
  const int larger_leaf_index = -1;
  CopyFromHostToCUDADevice<int>(cuda_leaf_index_, &larger_leaf_index, 1);
  SetCUDAMemory<double>(cuda_gain_, 0, 1);
  SetCUDAMemory<double>(cuda_leaf_value_, 0, 1);
  SynchronizeCUDADevice();
}

void CUDALeafSplits::InitValues(const data_size_t* cuda_data_indices_in_leaf, hist_t* cuda_hist_in_leaf,
    double* root_sum_hessians) {
  SetCUDAMemory<double>(cuda_sum_of_gradients_, 0, num_blocks_init_from_gradients_);
  SetCUDAMemory<double>(cuda_sum_of_hessians_, 0, num_blocks_init_from_gradients_);
  LaunchInitValuesKernal();
  SetCUDAMemory<int>(cuda_leaf_index_, 0, 1);
  CopyFromHostToCUDADeviceAsync<const data_size_t*>(cuda_data_indices_in_leaf_, &cuda_data_indices_in_leaf, 1, cuda_streams_[0]);
  CopyFromHostToCUDADeviceAsync<hist_t*>(cuda_hist_in_leaf_, &cuda_hist_in_leaf, 1, cuda_streams_[0]);
  CopyFromHostToCUDADeviceAsync<data_size_t>(cuda_num_data_in_leaf_, &num_data_, 1, cuda_streams_[0]);
  CopyFromCUDADeviceToHostAsync<double>(root_sum_hessians, cuda_sum_of_hessians_, 1, cuda_streams_[1]);
  SetCUDAMemory<double>(cuda_gain_, 0, 1);
  SetCUDAMemory<double>(cuda_leaf_value_, 0, 1);
  SynchronizeCUDADevice();
}

}  // namespace LightGBM

#endif  // USE_CUDA
