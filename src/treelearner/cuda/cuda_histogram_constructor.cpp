/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_histogram_constructor.hpp"

#include <chrono>

namespace LightGBM { 

CUDAHistogramConstructor::CUDAHistogramConstructor(const std::vector<int>& feature_group_ids,
  const Dataset* train_data, const int max_num_leaves,
  hist_t* cuda_hist): num_data_(train_data->num_data()),
  num_feature_groups_(feature_group_ids.size()),
  max_num_leaves_(max_num_leaves) {
  int offset = 0;
  int col_group_offset = 0;
  for (size_t i = 0; i < feature_group_ids.size(); ++i) {
    const int group_id = feature_group_ids[i];
    feature_group_bin_offsets_.emplace_back(offset);
    feature_group_bin_offsets_by_col_groups_.emplace_back(col_group_offset);
    offset += train_data->FeatureGroupNumBin(group_id);
    col_group_offset += train_data->FeatureGroupNumBin(group_id);
    if ((i + 1) % NUM_FEATURE_PER_THREAD_GROUP == 0) {
      col_group_offset = 0;
    }
  }
  feature_group_bin_offsets_.emplace_back(offset);
  feature_group_bin_offsets_by_col_groups_.emplace_back(col_group_offset);
  num_total_bin_ = offset;
  cuda_hist_ = cuda_hist;
}

void CUDAHistogramConstructor::Init() {
  // allocate CPU memory
  cpu_data_.resize(num_data_ * num_feature_groups_, 0);
  // allocate GPU memory
  void* cuda_data_ptr = nullptr;
  CUDASUCCESS_OR_FATAL(cudaMalloc(&cuda_data_ptr, num_data_ * num_feature_groups_ * sizeof(uint8_t)));
  cuda_data_ = reinterpret_cast<uint8_t*>(cuda_data_ptr);

  void* cuda_hist_ptr = nullptr;
  CUDASUCCESS_OR_FATAL(cudaMalloc(&cuda_hist_ptr, num_total_bin_ * max_num_leaves_ * sizeof(double)));
  cuda_hist_ = reinterpret_cast<hist_t*>(cuda_hist_ptr);

  void* cuda_num_total_bin_ptr = nullptr;
  CUDASUCCESS_OR_FATAL(cudaMalloc(&cuda_num_total_bin_ptr, sizeof(int)));
  cuda_num_total_bin_ = reinterpret_cast<int*>(cuda_num_total_bin_ptr);
  CUDASUCCESS_OR_FATAL(cudaMemcpy(cuda_num_total_bin_ptr, reinterpret_cast<const void*>(&num_total_bin_), sizeof(int), cudaMemcpyHostToDevice));

  AllocateCUDAMemory<int>(1, &cuda_num_feature_groups_);
  CopyFromHostToCUDADevice<int>(cuda_num_feature_groups_, &num_feature_groups_, 1);

  AllocateCUDAMemory<uint32_t>(feature_group_bin_offsets_.size(), &cuda_feature_group_bin_offsets_);
  CopyFromHostToCUDADevice<uint32_t>(cuda_feature_group_bin_offsets_,
    feature_group_bin_offsets_.data(),
    feature_group_bin_offsets_.size());
  AllocateCUDAMemory<uint32_t>(feature_group_bin_offsets_by_col_groups_.size(), &cuda_feature_group_bin_offsets_by_col_groups_);
  CopyFromHostToCUDADevice<uint32_t>(cuda_feature_group_bin_offsets_by_col_groups_,
    feature_group_bin_offsets_by_col_groups_.data(),
    feature_group_bin_offsets_by_col_groups_.size());
  /*for (size_t i = 0; i < feature_group_bin_offsets_.size(); ++i) {
    Log::Warning("feature_group_bin_offsets_[%d] = %d", i, feature_group_bin_offsets_[i]);
  }
  for (size_t i = 0; i < feature_group_bin_offsets_by_col_groups_.size(); ++i) {
    Log::Warning("feature_group_bin_offsets_by_col_groups_[%d] = %d", i, feature_group_bin_offsets_by_col_groups_[i]);
  }*/
}

void CUDAHistogramConstructor::PushOneData(const uint32_t feature_bin_value,
  const int feature_group_id,
  const data_size_t data_index) {
  const uint8_t feature_bin_value_uint8 = static_cast<uint8_t>(feature_bin_value);
  const size_t index = static_cast<size_t>(data_index) * static_cast<size_t>(num_feature_groups_) +
    static_cast<size_t>(feature_group_id);
  cpu_data_[index] = feature_bin_value_uint8;
}

void CUDAHistogramConstructor::FinishLoad() {
  // copy CPU data to GPU
  void* cuda_data_ptr = reinterpret_cast<void*>(cuda_data_);
  const void* cpu_data_ptr = reinterpret_cast<void*>(cpu_data_.data());
  CUDASUCCESS_OR_FATAL(cudaMemcpy(cuda_data_ptr, cpu_data_ptr, sizeof(uint8_t) * num_data_ * num_feature_groups_, cudaMemcpyHostToDevice));
}

void CUDAHistogramConstructor::ConstructHistogramForLeaf(const int* smaller_leaf_index, const int* /*larger_leaf_index*/,
  const data_size_t* num_data_in_leaf, const data_size_t* leaf_data_offset, const data_size_t* data_indices_ptr,
  const score_t* cuda_gradients, const score_t* cuda_hessians, const score_t* cuda_gradients_and_hessians) {
  //auto start = std::chrono::steady_clock::now();

  /*for (size_t i = 0; i < feature_group_bin_offsets_.size(); ++i) {
    Log::Warning("feature_group_bin_offsets_[%d] = %d", i, feature_group_bin_offsets_[i]);
  }*/
  AllocateCUDAMemory<int8_t>(num_data_, &cuda_int_gradients_);
  AllocateCUDAMemory<int8_t>(num_data_, &cuda_int_hessians_);
  AllocateCUDAMemory<int32_t>(num_data_, &cuda_int_gradients_and_hessians_);
  SetCUDAMemory<int8_t>(cuda_int_gradients_, 3, num_data_);
  SetCUDAMemory<int8_t>(cuda_int_hessians_, 3, num_data_);
  SetCUDAMemory<int32_t>(cuda_int_gradients_and_hessians_, 3, num_data_);
  /*for (size_t i = 0; i < feature_group_bin_offsets_by_col_groups_.size(); ++i) {
    Log::Warning("feature_group_bin_offsets_by_col_groups_[%d] = %d", i, feature_group_bin_offsets_by_col_groups_[i]);
  }*/
  auto start = std::chrono::steady_clock::now();
  LaunchConstructHistogramKernel(smaller_leaf_index, num_data_in_leaf, leaf_data_offset, data_indices_ptr,
    cuda_gradients, cuda_hessians, cuda_gradients_and_hessians);
  SynchronizeCUDADevice();
  auto end = std::chrono::steady_clock::now();
  double duration = (static_cast<std::chrono::duration<double>>(end - start)).count();
  Log::Warning("LaunchConstructHistogramKernel time %f", duration);
  PrintLastCUDAError();
  //Log::Warning("histogram construction finished");
  //Log::Warning("num_total_bin_ = %d", num_total_bin_);
  //Log::Warning("max_num_leaves_ = %d", max_num_leaves_);
  std::vector<hist_t> cpu_hist(6143 * 2, 0.0f);
  CopyFromCUDADeviceToHost<hist_t>(cpu_hist.data(), cuda_hist_, 6143 * 2);
  /*for (int i = 0; i < 6143; ++i) {
    Log::Warning("bin %d grad %f hess %f", i, cpu_hist[2 * i], cpu_hist[2 * i + 1]);
  }*/
}



}  // namespace LightGBM

#endif  // USE_CUDA
