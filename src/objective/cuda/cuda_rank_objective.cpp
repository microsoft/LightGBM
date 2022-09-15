/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA_EXP

#include <string>
#include <vector>

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
    cuda_item_indices_buffer_.Resize(static_cast<size_t>(metadata.query_boundaries()[metadata.num_queries()]));
  }
  cuda_labels_ = metadata.cuda_metadata()->cuda_label();
  cuda_query_boundaries_ = metadata.cuda_metadata()->cuda_query_boundaries();
  cuda_inverse_max_dcgs_.Resize(inverse_max_dcgs_.size());
  CopyFromHostToCUDADevice(cuda_inverse_max_dcgs_.RawData(), inverse_max_dcgs_.data(), inverse_max_dcgs_.size(), __FILE__, __LINE__);
  cuda_label_gain_.Resize(label_gain_.size());
  CopyFromHostToCUDADevice(cuda_label_gain_.RawData(), label_gain_.data(), label_gain_.size(), __FILE__, __LINE__);
}

void CUDALambdarankNDCG::GetGradients(const double* score, score_t* gradients, score_t* hessians) const {
  LaunchGetGradientsKernel(score, gradients, hessians);
}


CUDARankXENDCG::CUDARankXENDCG(const Config& config): CUDALambdarankNDCG(config) {}

CUDARankXENDCG::CUDARankXENDCG(const std::vector<std::string>& strs): CUDALambdarankNDCG(strs) {}

CUDARankXENDCG::~CUDARankXENDCG() {}

void CUDARankXENDCG::Init(const Metadata& metadata, data_size_t num_data) {
  CUDALambdarankNDCG::Init(metadata, num_data);
  for (data_size_t i = 0; i < num_queries_; ++i) {
    rands_.emplace_back(seed_ + i);
  }
  item_rands_.resize(num_data, 0.0f);
  AllocateCUDAMemory<double>(&cuda_item_rands_, static_cast<size_t>(num_data), __FILE__, __LINE__);
  if (max_items_in_query_aligned_ >= 2048) {
    AllocateCUDAMemory<double>(&cuda_params_buffer_, static_cast<size_t>(num_data_), __FILE__, __LINE__);
  }
}

void CUDARankXENDCG::GenerateItemRands() const {
  const int num_threads = OMP_NUM_THREADS();
  OMP_INIT_EX();
  #pragma omp parallel for schedule(static) num_threads(num_threads)
  for (data_size_t i = 0; i < num_queries_; ++i) {
    OMP_LOOP_EX_BEGIN();
    const data_size_t start = query_boundaries_[i];
    const data_size_t end = query_boundaries_[i + 1];
    for (data_size_t j = start; j < end; ++j) {
      item_rands_[j] = rands_[i].NextFloat();
    }
    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();
}

void CUDARankXENDCG::GetGradients(const double* score, score_t* gradients, score_t* hessians) const {
  GenerateItemRands();
  CopyFromHostToCUDADevice<double>(cuda_item_rands_, item_rands_.data(), item_rands_.size(), __FILE__, __LINE__);
  LaunchGetGradientsKernel(score, gradients, hessians);
}


}  // namespace LightGBM

#endif  // USE_CUDA_EXP
