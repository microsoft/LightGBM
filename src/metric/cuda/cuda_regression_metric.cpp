/*!
 * Copyright (c) 2022 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA_EXP

#include <vector>

#include "cuda_regression_metric.hpp"

namespace LightGBM {

template <typename HOST_METRIC, typename CUDA_METRIC>
void CUDARegressionMetricInterface<HOST_METRIC, CUDA_METRIC>::Init(const Metadata& metadata, data_size_t num_data) {
  CUDAMetricInterface<HOST_METRIC>::Init(metadata, num_data);
  const int max_num_reduce_blocks = (this->num_data_ + NUM_DATA_PER_EVAL_THREAD - 1) / NUM_DATA_PER_EVAL_THREAD;
  if (this->cuda_weights_ == nullptr) {
    reduce_block_buffer_.Resize(max_num_reduce_blocks);
  } else {
    reduce_block_buffer_.Resize(max_num_reduce_blocks * 2);
  }
  const int max_num_reduce_blocks_inner = (max_num_reduce_blocks + NUM_DATA_PER_EVAL_THREAD - 1) / NUM_DATA_PER_EVAL_THREAD;
  if (this->cuda_weights_ == nullptr) {
    reduce_block_buffer_inner_.Resize(max_num_reduce_blocks_inner);
  } else {
    reduce_block_buffer_inner_.Resize(max_num_reduce_blocks_inner * 2);
  }
}

template <typename HOST_METRIC, typename CUDA_METRIC>
std::vector<double> CUDARegressionMetricInterface<HOST_METRIC, CUDA_METRIC>::Eval(const double* score, const ObjectiveFunction* objective) const {
  if (objective->NeedConvertOutputCUDA()) {
    score_convert_buffer_.Resize(static_cast<size_t>(this->num_data_) * static_cast<size_t>(this->num_class_));
  }
  const double* score_convert = objective->ConvertOutputCUDA(this->num_data_, score, score_convert_buffer_.RawData());
  const double eval_score = LaunchEvalKernel(score_convert);
  return std::vector<double>{eval_score};
}

CUDARMSEMetric::CUDARMSEMetric(const Config& config): CUDARegressionMetricInterface<RMSEMetric, CUDARMSEMetric>(config) {}

CUDAL2Metric::CUDAL2Metric(const Config& config): CUDARegressionMetricInterface<L2Metric, CUDAL2Metric>(config) {}

}  // namespace LightGBM

#endif  // USE_CUDA_EXP
