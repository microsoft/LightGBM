/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_ranking_objective.hpp"

namespace LightGBM {

CUDARankingObjective::CUDARankingObjective(
  const data_size_t num_data,
  const label_t* cuda_labels,
  const data_size_t* cuda_query_boundaries,
  const data_size_t* cpu_query_boundaries,
  const int num_queries,
  const bool norm,
  const double sigmoid,
  const int truncation_level,
  const label_t* labels,
  const int num_threads):
CUDAObjective(num_data),
cuda_labels_(cuda_labels),
cuda_query_boundaries_(cuda_query_boundaries),
num_queries_(num_queries),
norm_(norm),
sigmoid_(sigmoid),
truncation_level_(truncation_level),
num_threads_(num_threads) {
  std::vector<label_t> thread_max_label(num_threads, 0.0f);
  Threading::For<data_size_t>(0, num_data_, 512,
    [labels, &thread_max_label, this] (int thread_index, data_size_t start, data_size_t end) {
      if (start < num_data_) {
        thread_max_label[thread_index] = labels[start];
      }
      for (data_size_t data_index = start + 1; data_index < end; ++data_index) {
        const label_t label = labels[data_index];
        if (label > thread_max_label[thread_index]) {
          thread_max_label[thread_index] = label;
        }
      }
    });
  max_label_ = thread_max_label[0];
  for (int thread_index = 1; thread_index < num_threads_; ++thread_index) {
    max_label_ = std::max(max_label_, thread_max_label[thread_index]);
  }

  std::vector<uint16_t> thread_max_num_items_in_query(num_threads_);
  Threading::For<data_size_t>(0, num_queries_, 1,
    [cpu_query_boundaries, &thread_max_num_items_in_query] (int thread_index, data_size_t start, data_size_t end) {
      for (data_size_t query_index = start; query_index < end; ++query_index) {
        const data_size_t query_item_count = cpu_query_boundaries[query_index + 1] - cpu_query_boundaries[query_index];
        if (query_item_count > thread_max_num_items_in_query[thread_index]) {
          thread_max_num_items_in_query[thread_index] = query_item_count;
        }
      }
    });
  data_size_t max_items_in_query = 0;
  for (int thread_index = 0; thread_index < num_threads_; ++thread_index) {
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
  if (max_items_in_query_aligned_ > MAX_NUM_ITEM_IN_QUERY) {
    Log::Warning("Too many items in a query.");
  }
}

void CUDARankingObjective::Init() {
  AllocateCUDAMemory<double>(1, &cuda_init_score_);
  SetCUDAMemory<double>(cuda_init_score_, 0, 1);
  AllocateCUDAMemory<double>(num_data_, &cuda_lambdas_);
  AllocateCUDAMemory<double>(num_queries_, &cuda_inverse_max_dcgs_);
  LaunchCalcInverseMaxDCGKernel();
}

void CUDARankingObjective::GetGradients(const double* cuda_scores, score_t* cuda_out_gradients, score_t* cuda_out_hessians) {
  LaunchGetGradientsKernel(cuda_scores, cuda_out_gradients, cuda_out_hessians);
}

void CUDARankingObjective::CalcInitScore() {}

void CUDARankingObjective::TestGlobalArgSort() const {
  LaunchGlobalArgSort();
}

}  // namespace LightGBM

#endif  // USE_CUDA
