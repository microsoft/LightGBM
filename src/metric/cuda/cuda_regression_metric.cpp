/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include "cuda_regression_metric.hpp"

namespace LightGBM {

template <typename CUDAPointWiseLossCalculator>
CUDARegressionMetric<CUDAPointWiseLossCalculator>::CUDARegressionMetric(const Config& config): RegressionMetric<CUDAPointWiseLossCalculator>(config) {}

template <typename CUDAPointWiseLossCalculator>
CUDARegressionMetric<CUDAPointWiseLossCalculator>::~CUDARegressionMetric() {}

template <typename CUDAPointWiseLossCalculator>
void CUDARegressionMetric<CUDAPointWiseLossCalculator>::Init(const Metadata& metadata, data_size_t num_data) {
  RegressionMetric<CUDAPointWiseLossCalculator>::Init(metadata, num_data);
  cuda_label_ = metadata.cuda_metadata()->cuda_label();
  cuda_weights_ = metadata.cuda_metadata()->cuda_weights();

  const data_size_t num_blocks = (num_data + EVAL_BLOCK_SIZE_REGRESSION_METRIC - 1) / EVAL_BLOCK_SIZE_REGRESSION_METRIC;
  AllocateCUDAMemoryOuter<double>(&cuda_sum_loss_buffer_, static_cast<size_t>(num_blocks), __FILE__, __LINE__);
  AllocateCUDAMemoryOuter<double>(&cuda_score_convert_buffer_, static_cast<size_t>(num_data), __FILE__, __LINE__);
  AllocateCUDAMemoryOuter<double>(&cuda_sum_loss_, 1, __FILE__, __LINE__);
}

template <typename CUDAPointWiseLossCalculator>
std::vector<double> CUDARegressionMetric<CUDAPointWiseLossCalculator>::Eval(const double* score, const ObjectiveFunction* objective) const {
  double sum_loss = 0.0f;
  objective->GetCUDAConvertOutputFunc()(this->num_data_, score, cuda_score_convert_buffer_);
  LaunchEvalKernel(cuda_score_convert_buffer_);
  CopyFromCUDADeviceToHostOuter<double>(&sum_loss, cuda_sum_loss_, 1, __FILE__, __LINE__);
  return std::vector<double>(1, CUDAPointWiseLossCalculator::AverageLoss(sum_loss, this->sum_weights_));
}

CUDARMSEMetric::CUDARMSEMetric(const Config& config): CUDARegressionMetric<CUDARMSEMetric>(config) {}

CUDAL2Metric::CUDAL2Metric(const Config& config): CUDARegressionMetric<CUDAL2Metric>(config) {}

CUDAL1Metric::CUDAL1Metric(const Config& config): CUDARegressionMetric<CUDAL1Metric>(config) {}

CUDAQuantileMetric::CUDAQuantileMetric(const Config& config): CUDARegressionMetric<CUDAQuantileMetric>(config) {}

CUDAHuberLossMetric::CUDAHuberLossMetric(const Config& config): CUDARegressionMetric<CUDAHuberLossMetric>(config) {}

CUDAFairLossMetric::CUDAFairLossMetric(const Config& config): CUDARegressionMetric<CUDAFairLossMetric>(config) {}

CUDAPoissonMetric::CUDAPoissonMetric(const Config& config): CUDARegressionMetric<CUDAPoissonMetric>(config) {}

CUDAMAPEMetric::CUDAMAPEMetric(const Config& config): CUDARegressionMetric<CUDAMAPEMetric>(config) {}

CUDAGammaMetric::CUDAGammaMetric(const Config& config): CUDARegressionMetric<CUDAGammaMetric>(config) {}

CUDAGammaDevianceMetric::CUDAGammaDevianceMetric(const Config& config): CUDARegressionMetric<CUDAGammaDevianceMetric>(config) {}

CUDATweedieMetric::CUDATweedieMetric(const Config& config): CUDARegressionMetric<CUDATweedieMetric>(config) {}

}  // namespace LightGBM
