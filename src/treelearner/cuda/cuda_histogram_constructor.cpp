/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_histogram_constructor.hpp"

namespace LightGBM { 

CUDAHistogramConstructor::CUDAHistogramConstructor(const Dataset* train_data,
  const int num_leaves, const int num_threads,
  const score_t* cuda_gradients, const score_t* cuda_hessians): num_data_(train_data->num_data()),
  num_features_(train_data->num_features()), num_leaves_(num_leaves), num_threads_(num_threads),
  num_feature_groups_(train_data->num_feature_groups()),
  cuda_gradients_(cuda_gradients), cuda_hessians_(cuda_hessians) {
  int offset = 0;
  for (int group_id = 0; group_id < train_data->num_feature_groups(); ++group_id) {
    feature_group_bin_offsets_.emplace_back(offset);
    offset += train_data->FeatureGroupNumBin(group_id);
  }
  feature_group_bin_offsets_.emplace_back(offset);
  num_total_bin_ = offset;
}

void CUDAHistogramConstructor::Init(const Dataset* train_data) {
  // allocate CPU memory
  data_.resize(num_data_ * num_feature_groups_, 0);
  // allocate GPU memory
  AllocateCUDAMemory<uint8_t>(num_feature_groups_ * num_data_, &cuda_data_);

  AllocateCUDAMemory<hist_t>(num_total_bin_ * 2 * num_leaves_, &cuda_hist_);

  InitCUDAMemoryFromHostMemory<int>(&cuda_num_total_bin_, &num_total_bin_, 1);

  InitCUDAMemoryFromHostMemory<int>(&cuda_num_feature_groups_, &num_feature_groups_, 1);

  InitCUDAMemoryFromHostMemory<uint32_t>(&cuda_feature_group_bin_offsets_,
    feature_group_bin_offsets_.data(), feature_group_bin_offsets_.size());

  InitCUDAData(train_data);
}

void CUDAHistogramConstructor::InitCUDAData(const Dataset* train_data) {
  std::vector<std::unique_ptr<BinIterator>> bin_iterators(num_feature_groups_);
  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (int group_id = 0; group_id < num_feature_groups_; ++group_id) {
    bin_iterators[group_id].reset(train_data->FeatureGroupIterator(group_id));
    bin_iterators[group_id]->Reset(0);
    for (data_size_t data_index = 0; data_index < num_data_; ++data_index) {
      const uint32_t bin = static_cast<uint8_t>(bin_iterators[group_id]->RawGet(data_index));
      PushOneData(bin, group_id, data_index);
    }
  }
  CopyFromHostToCUDADevice<uint8_t>(cuda_data_, data_.data(), data_.size());
  SynchronizeCUDADevice();
}

void CUDAHistogramConstructor::PushOneData(const uint32_t feature_bin_value,
  const int feature_group_id,
  const data_size_t data_index) {
  const uint8_t feature_bin_value_uint8 = static_cast<uint8_t>(feature_bin_value);
  const size_t index = static_cast<size_t>(data_index) * static_cast<size_t>(num_feature_groups_) +
    static_cast<size_t>(feature_group_id);
  data_[index] = feature_bin_value_uint8;
}

void CUDAHistogramConstructor::ConstructHistogramForLeaf(const int* cuda_smaller_leaf_index, const int* /*cuda_larger_leaf_index*/,
  const data_size_t* cuda_data_indices_in_smaller_leaf, const data_size_t* /*cuda_data_indices_in_larger_leaf*/,
  const data_size_t* cuda_leaf_num_data) {
  auto start = std::chrono::steady_clock::now();
  LaunchConstructHistogramKernel(cuda_smaller_leaf_index, cuda_data_indices_in_smaller_leaf, cuda_leaf_num_data);
  SynchronizeCUDADevice();
  auto end = std::chrono::steady_clock::now();
  double duration = (static_cast<std::chrono::duration<double>>(end - start)).count();
  Log::Warning("LaunchConstructHistogramKernel time %f", duration);
  /*PrintLastCUDAError();
  std::vector<hist_t> cpu_hist(6143 * 2, 0.0f);
  CopyFromCUDADeviceToHost<hist_t>(cpu_hist.data(), cuda_hist_, 6143 * 2);*/
}

}  // namespace LightGBM

#endif  // USE_CUDA
