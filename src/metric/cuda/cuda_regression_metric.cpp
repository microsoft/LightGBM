/*!
 * Copyright (c) 2022 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include <vector>

#include "cuda_regression_metric.hpp"

namespace LightGBM {

template <typename HOST_METRIC, typename CUDA_METRIC>
std::vector<double> CUDARegressionMetricInterface<HOST_METRIC, CUDA_METRIC>::Eval(const double* score, const ObjectiveFunction* objective) const {
  const double* score_convert = score;
  if (objective != nullptr && objective->NeedConvertOutputCUDA()) {
    this->score_convert_buffer_.Resize(static_cast<size_t>(this->num_data_) * static_cast<size_t>(this->num_class_));
    score_convert = objective->ConvertOutputCUDA(this->num_data_, score, this->score_convert_buffer_.RawData());
  }
  double sum_loss = 0.0, sum_weight = 0.0;
  this->LaunchEvalKernel(score_convert, &sum_loss, &sum_weight);
  const double eval_score = this->AverageLoss(sum_loss, sum_weight);
  return std::vector<double>{eval_score};
}

CUDARMSEMetric::CUDARMSEMetric(const Config& config): CUDARegressionMetricInterface<RMSEMetric, CUDARMSEMetric>(config) {}

CUDAL2Metric::CUDAL2Metric(const Config& config): CUDARegressionMetricInterface<L2Metric, CUDAL2Metric>(config) {}

CUDAQuantileMetric::CUDAQuantileMetric(const Config& config): CUDARegressionMetricInterface<QuantileMetric, CUDAQuantileMetric>(config), alpha_(config.alpha) {}

}  // namespace LightGBM

#endif  // USE_CUDA
