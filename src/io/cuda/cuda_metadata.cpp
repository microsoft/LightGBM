/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifdef USE_CUDA

#include <LightGBM/cuda/cuda_metadata.hpp>

namespace LightGBM {

CUDAMetadata::CUDAMetadata(const int gpu_device_id) {
  if (gpu_device_id >= 0) {
    SetCUDADevice(gpu_device_id, __FILE__, __LINE__);
  } else {
    SetCUDADevice(0, __FILE__, __LINE__);
  }
  cuda_label_ = nullptr;
  cuda_weights_ = nullptr;
  cuda_query_boundaries_ = nullptr;
  cuda_query_weights_ = nullptr;
  cuda_init_score_ = nullptr;
}

CUDAMetadata::~CUDAMetadata() {
  DeallocateCUDAMemory<label_t>(&cuda_label_, __FILE__, __LINE__);
  DeallocateCUDAMemory<label_t>(&cuda_weights_, __FILE__, __LINE__);
  DeallocateCUDAMemory<data_size_t>(&cuda_query_boundaries_, __FILE__, __LINE__);
  DeallocateCUDAMemory<label_t>(&cuda_query_weights_, __FILE__, __LINE__);
  DeallocateCUDAMemory<double>(&cuda_init_score_, __FILE__, __LINE__);
}

void CUDAMetadata::Init(const std::vector<label_t>& label,
                        const std::vector<label_t>& weight,
                        const std::vector<data_size_t>& query_boundaries,
                        const std::vector<label_t>& query_weights,
                        const std::vector<double>& init_score) {
  if (label.size() == 0) {
    cuda_label_ = nullptr;
  } else {
    InitCUDAMemoryFromHostMemory<label_t>(&cuda_label_, label.data(), label.size(), __FILE__, __LINE__);
  }
  if (weight.size() == 0) {
    cuda_weights_ = nullptr;
  } else {
    InitCUDAMemoryFromHostMemory<label_t>(&cuda_weights_, weight.data(), weight.size(), __FILE__, __LINE__);
  }
  if (query_boundaries.size() == 0) {
    cuda_query_boundaries_ = nullptr;
  } else {
    InitCUDAMemoryFromHostMemory<data_size_t>(&cuda_query_boundaries_, query_boundaries.data(), query_boundaries.size(), __FILE__, __LINE__);
  }
  if (query_weights.size() == 0) {
    cuda_query_weights_ = nullptr;
  } else {
    InitCUDAMemoryFromHostMemory<label_t>(&cuda_query_weights_, query_weights.data(), query_weights.size(), __FILE__, __LINE__);
  }
  if (init_score.size() == 0) {
    cuda_init_score_ = nullptr;
  } else {
    InitCUDAMemoryFromHostMemory<double>(&cuda_init_score_, init_score.data(), init_score.size(), __FILE__, __LINE__);
  }
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

void CUDAMetadata::SetLabel(const label_t* label, data_size_t len) {
  DeallocateCUDAMemory<label_t>(&cuda_label_, __FILE__, __LINE__);
  InitCUDAMemoryFromHostMemory<label_t>(&cuda_label_, label, static_cast<size_t>(len), __FILE__, __LINE__);
}

void CUDAMetadata::SetWeights(const label_t* weights, data_size_t len) {
  DeallocateCUDAMemory<label_t>(&cuda_weights_, __FILE__, __LINE__);
  InitCUDAMemoryFromHostMemory<label_t>(&cuda_weights_, weights, static_cast<size_t>(len), __FILE__, __LINE__);
}

void CUDAMetadata::SetQuery(const data_size_t* query_boundaries, const label_t* query_weights, data_size_t num_queries) {
  DeallocateCUDAMemory<data_size_t>(&cuda_query_boundaries_, __FILE__, __LINE__);
  InitCUDAMemoryFromHostMemory<data_size_t>(&cuda_query_boundaries_, query_boundaries, static_cast<size_t>(num_queries) + 1, __FILE__, __LINE__);
  if (query_weights != nullptr) {
    DeallocateCUDAMemory<label_t>(&cuda_query_weights_, __FILE__, __LINE__);
    InitCUDAMemoryFromHostMemory<label_t>(&cuda_query_weights_, query_weights, static_cast<size_t>(num_queries), __FILE__, __LINE__);
  }
}

void CUDAMetadata::SetInitScore(const double* init_score, data_size_t len) {
  DeallocateCUDAMemory<double>(&cuda_init_score_, __FILE__, __LINE__);
  InitCUDAMemoryFromHostMemory<double>(&cuda_init_score_, init_score, static_cast<size_t>(len), __FILE__, __LINE__);
}

}  // namespace LightGBM

#endif  // USE_CUDA
