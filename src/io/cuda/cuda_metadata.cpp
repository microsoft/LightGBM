/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifdef USE_CUDA

#include <LightGBM/cuda/cuda_metadata.hpp>

namespace LightGBM {

CUDAMetadata::CUDAMetadata(const int gpu_device_id) {
  if (gpu_device_id >= 0) {
    CUDASUCCESS_OR_FATAL(cudaSetDevice(gpu_device_id));
  } else {
    CUDASUCCESS_OR_FATAL(cudaSetDevice(0));
  }
  cuda_label_ = nullptr;
  cuda_weights_ = nullptr;
  cuda_query_boundaries_ = nullptr;
  cuda_query_weights_ = nullptr;
  cuda_init_score_ = nullptr;
  cuda_queries_ = nullptr;
}

CUDAMetadata::~CUDAMetadata() {
  DeallocateCUDAMemory<label_t>(&cuda_label_, __FILE__, __LINE__);
  DeallocateCUDAMemory<label_t>(&cuda_weights_, __FILE__, __LINE__);
  DeallocateCUDAMemory<data_size_t>(&cuda_query_boundaries_, __FILE__, __LINE__);
  DeallocateCUDAMemory<label_t>(&cuda_query_weights_, __FILE__, __LINE__);
  DeallocateCUDAMemory<double>(&cuda_init_score_, __FILE__, __LINE__);
  DeallocateCUDAMemory<data_size_t>(&cuda_queries_, __FILE__, __LINE__);
}

void CUDAMetadata::Init(const std::vector<label_t>& label,
                        const std::vector<label_t>& weight,
                        const std::vector<data_size_t>& query_boundaries,
                        const std::vector<label_t>& query_weights,
                        const std::vector<double>& init_score,
                        const std::vector<data_size_t>& queries) {
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
  if (queries.size() == 0) {
    cuda_queries_ = nullptr;
  } else {
    InitCUDAMemoryFromHostMemory<data_size_t>(&cuda_queries_, queries.data(), queries.size(), __FILE__, __LINE__);
  }
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

}  // namespace LightGBM

#endif  // USE_CUDA
