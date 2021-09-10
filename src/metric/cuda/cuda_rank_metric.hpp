/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_METRIC_CUDA_CUDA_RANK_METRIC_HPP_
#define LIGHTGBM_METRIC_CUDA_CUDA_RANK_METRIC_HPP_

#include "cuda_metric.hpp"
#include "../rank_metric.hpp"

#define EVAL_BLOCK_SIZE_RANK_METRIC (1024)
#define NUM_QUERY_PER_BLOCK_METRIC (10)
#define MAX_RANK_LABEL_METRIC (32)

namespace LightGBM {

class CUDANDCGMetric : public CUDAMetricInterface, public NDCGMetric {
 public:
  explicit CUDANDCGMetric(const Config& config);

  ~CUDANDCGMetric();

  void Init(const Metadata& metadata, data_size_t num_data) override;

  std::vector<double> Eval(const double* score, const ObjectiveFunction*) const override;

 private:
  void LaunchEvalKernel(const double* score) const;

  const label_t* cuda_label_;
  const label_t* cuda_weights_;
  const data_size_t* cuda_query_boundaries_;
  const label_t* cuda_query_weights_;
  data_size_t* cuda_eval_at_;
  double* cuda_inverse_max_dcgs_;
  double* cuda_label_gain_;
  double* cuda_discount_;
  double* cuda_block_dcg_buffer_;
  double* cuda_ndcg_result_;
  data_size_t* cuda_item_indices_buffer_;
  int max_items_in_query_aligned_;
  int max_items_in_query_;
  int num_eval_;
};

}  // namespace LightGBM

#endif  // LIGHTGBM_METRIC_CUDA_CUDA_RANK_METRIC_HPP_
