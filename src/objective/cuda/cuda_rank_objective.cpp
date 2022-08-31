/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA_EXP

#include "cuda_rank_objective.hpp"

namespace LightGBM {

CUDALambdarankNDCG::CUDALambdarankNDCG(const Config& config):
LambdarankNDCG(config) {}

CUDALambdarankNDCG::CUDALambdarankNDCG(const std::vector<std::string>& strs): LambdarankNDCG(strs) {}

void CUDALambdarankNDCG::Init(const Metadata& metadata, data_size_t num_data) {
  const int num_threads = OMP_NUM_THREADS();
  LambdarankNDCG::Init(metadata, num_data);

  std::vector<uint16_t> thread_max_num_items_in_query(num_threads);
  Threading::For<data_size_t>(0, num_queries_, 1,
    [this, &thread_max_num_items_in_query] (int thread_index, data_size_t start, data_size_t end) {
      for (data_size_t query_index = start; query_index < end; ++query_index) {
        const data_size_t query_item_count = query_boundaries_[query_index + 1] - query_boundaries_[query_index];
        if (query_item_count > thread_max_num_items_in_query[thread_index]) {
          thread_max_num_items_in_query[thread_index] = query_item_count;
        }
      }
    });
  data_size_t max_items_in_query = 0;
  for (int thread_index = 0; thread_index < num_threads; ++thread_index) {
    if (thread_max_num_items_in_query[thread_index] > max_items_in_query) {
      max_items_in_query = thread_max_num_items_in_query[thread_index];
    }
  }
  max_items_in_query_aligned_ = 1;
  --max_items_in_query;
  while (max_items_in_query > 0) {
    max_items_in_query >>= 1;
    max_items_in_query_aligned_ <<= 1;
  }
  if (max_items_in_query_aligned_ > 2048) {
    AllocateCUDAMemory<int>(&cuda_item_indices_buffer_,
                                 static_cast<size_t>(metadata.query_boundaries()[metadata.num_queries()]),
                                 __FILE__, __LINE__);
  }
  cuda_labels_ = metadata.cuda_metadata()->cuda_label();
  cuda_query_boundaries_ = metadata.cuda_metadata()->cuda_query_boundaries();
  AllocateCUDAMemory<double>(&cuda_lambdas_, num_data_, __FILE__, __LINE__);
  AllocateCUDAMemory<double>(&cuda_inverse_max_dcgs_, num_queries_, __FILE__, __LINE__);
  LaunchCalcInverseMaxDCGKernel();
}

void CUDALambdarankNDCG::GetGradients(const double* score, score_t* gradients, score_t* hessians) const {
  LaunchGetGradientsKernel(score, gradients, hessians);
}


}  // namespace LightGBM

#endif  // USE_CUDA_EXP
