/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include "cuda_rank_metric.hpp"

namespace LightGBM {

CUDANDCGMetric::CUDANDCGMetric(const Config& config): NDCGMetric(config) {}

CUDANDCGMetric::~CUDANDCGMetric() {}

void CUDANDCGMetric::Init(const Metadata& metadata, data_size_t num_data) {
  NDCGMetric::Init(metadata, num_data);
  cuda_label_ = metadata.cuda_metadata()->cuda_label();
  cuda_weights_ = metadata.cuda_metadata()->cuda_weights();
  cuda_query_boundaries_ = metadata.cuda_metadata()->cuda_query_boundaries();
  cuda_query_weights_ = metadata.cuda_metadata()->cuda_query_weights();
  const int num_threads = OMP_NUM_THREADS();
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
  max_items_in_query_ = 0;
  for (int thread_index = 0; thread_index < num_threads; ++thread_index) {
    if (thread_max_num_items_in_query[thread_index] > max_items_in_query_) {
      max_items_in_query_ = thread_max_num_items_in_query[thread_index];
    }
  }
  max_items_in_query_aligned_ = 1;
  --max_items_in_query_;
  while (max_items_in_query_ > 0) {
    max_items_in_query_ >>= 1;
    max_items_in_query_aligned_ <<= 1;
  }
  num_eval_ = static_cast<data_size_t>(eval_at_.size());
  InitCUDAMemoryFromHostMemoryOuter<data_size_t>(&cuda_eval_at_, eval_at_.data(), eval_at_.size(), __FILE__, __LINE__);
  const size_t total_inverse_max_dcg_items = static_cast<size_t>(num_queries_ * num_eval_);
  std::vector<double> flatten_inverse_max_dcgs(total_inverse_max_dcg_items, 0.0f);
  OMP_INIT_EX();
  #pragma omp parallel for schedule(static) num_threads(num_threads)
  for (data_size_t query_index = 0; query_index < num_queries_; ++query_index) {
    OMP_LOOP_EX_BEGIN();
    for (data_size_t eval_index = 0; eval_index < num_eval_; ++eval_index) {
      flatten_inverse_max_dcgs[query_index * num_eval_ + eval_index] = inverse_max_dcgs_[query_index][eval_index];
    }
    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();
  InitCUDAMemoryFromHostMemoryOuter<double>(&cuda_inverse_max_dcgs_, flatten_inverse_max_dcgs.data(), flatten_inverse_max_dcgs.size(), __FILE__, __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<double>(&cuda_label_gain_, DCGCalculator::label_gain().data(), DCGCalculator::label_gain().size(), __FILE__, __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<double>(&cuda_discount_, DCGCalculator::discount().data(), DCGCalculator::discount().size(), __FILE__, __LINE__);
  const int num_blocks = (num_queries_ + NUM_QUERY_PER_BLOCK_METRIC - 1) / NUM_QUERY_PER_BLOCK_METRIC;
  AllocateCUDAMemoryOuter<double>(&cuda_block_dcg_buffer_, static_cast<size_t>(num_blocks * num_eval_), __FILE__, __LINE__);
  AllocateCUDAMemoryOuter<data_size_t>(&cuda_item_indices_buffer_, static_cast<size_t>(num_data_), __FILE__, __LINE__);
  AllocateCUDAMemoryOuter<double>(&cuda_ndcg_result_, static_cast<size_t>(num_eval_), __FILE__, __LINE__);
}

std::vector<double> CUDANDCGMetric::Eval(const double* score, const ObjectiveFunction*) const {
  LaunchEvalKernel(score);
  std::vector<double> result(num_eval_, 0.0f);
  CopyFromCUDADeviceToHostOuter<double>(result.data(), cuda_ndcg_result_, static_cast<size_t>(num_eval_), __FILE__, __LINE__);
  return result;
}

}  // namespace LightGBM
