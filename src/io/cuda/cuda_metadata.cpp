/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <LightGBM/cuda/cuda_metadata.hpp>

namespace LightGBM {

CUDAMetadata::CUDAMetadata() {}

CUDAMetadata::~CUDAMetadata() {}

void CUDAMetadata::Init(const std::vector<label_t>& label,
                        const std::vector<label_t>& weight,
                        const std::vector<data_size_t>& query_boundaries,
                        const std::vector<label_t>& query_weights,
                        const std::vector<double>& init_score,
                        const std::vector<data_size_t>& queries) {
  Log::Warning("label.size() = %d", label.size());
  InitCUDAMemoryFromHostMemoryOuter<label_t>(&cuda_label_, label.data(), label.size(), __FILE__, __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<label_t>(&cuda_weights_, weight.data(), weight.size(), __FILE__, __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<data_size_t>(&cuda_query_boundaries_, query_boundaries.data(), query_boundaries.size(), __FILE__, __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<label_t>(&cuda_query_weights_, query_weights.data(), query_weights.size(), __FILE__, __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<double>(&cuda_init_score_, init_score.data(), init_score.size(), __FILE__, __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<data_size_t>(&cuda_queries_, queries.data(), queries.size(), __FILE__, __LINE__);
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
}

} // namespace LightGBM
