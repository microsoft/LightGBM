/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include "cuda_xentropy_metric.hpp"

namespace LightGBM {

CUDACrossEntropyMetric::CUDACrossEntropyMetric(const Config& config): CrossEntropyMetric(config) {}

CUDACrossEntropyMetric::~CUDACrossEntropyMetric() {}

void CUDACrossEntropyMetric::Init(const Metadata& metadata, data_size_t num_data) {
  CrossEntropyMetric::Init(metadata, num_data);
  cuda_label_ = metadata.cuda_metadata()->cuda_label();
  cuda_weights_ = metadata.cuda_metadata()->cuda_weights();
  AllocateCUDAMemoryOuter<double>(&cuda_score_convert_buffer_, static_cast<size_t>(num_data_), __FILE__, __LINE__);

  const data_size_t num_blocks = (num_data + EVAL_BLOCK_SIZE_XENTROPY_METRIC - 1) / EVAL_BLOCK_SIZE_XENTROPY_METRIC;
  AllocateCUDAMemoryOuter<double>(&cuda_sum_loss_buffer_, static_cast<size_t>(num_blocks), __FILE__, __LINE__);
  AllocateCUDAMemoryOuter<double>(&cuda_sum_loss_, 1, __FILE__, __LINE__);
}

std::vector<double> CUDACrossEntropyMetric::Eval(const double* score, const ObjectiveFunction* objective) const {
  double sum_loss = 0.0f;
  objective->GetCUDAConvertOutputFunc()(num_data_, score, cuda_score_convert_buffer_);
  LaunchEvalKernel(cuda_score_convert_buffer_);
  CopyFromCUDADeviceToHostOuter<double>(&sum_loss, cuda_sum_loss_, 1, __FILE__, __LINE__);
  return std::vector<double>(1, sum_loss / sum_weights_);
}

CUDACrossEntropyLambdaMetric::CUDACrossEntropyLambdaMetric(const Config& config): CrossEntropyLambdaMetric(config) {}

CUDACrossEntropyLambdaMetric::~CUDACrossEntropyLambdaMetric() {}

void CUDACrossEntropyLambdaMetric::Init(const Metadata& metadata, data_size_t num_data) {
  CrossEntropyLambdaMetric::Init(metadata, num_data);
  cuda_label_ = metadata.cuda_metadata()->cuda_label();
  cuda_weights_ = metadata.cuda_metadata()->cuda_weights();
  AllocateCUDAMemoryOuter<double>(&cuda_score_convert_buffer_, static_cast<size_t>(num_data_), __FILE__, __LINE__);

  const data_size_t num_blocks = (num_data + EVAL_BLOCK_SIZE_XENTROPY_METRIC - 1) / EVAL_BLOCK_SIZE_XENTROPY_METRIC;
  AllocateCUDAMemoryOuter<double>(&cuda_sum_loss_buffer_, static_cast<size_t>(num_blocks), __FILE__, __LINE__);
  AllocateCUDAMemoryOuter<double>(&cuda_sum_loss_, 1, __FILE__, __LINE__);
}

std::vector<double> CUDACrossEntropyLambdaMetric::Eval(const double* score, const ObjectiveFunction* objective) const {
  objective->GetCUDAConvertOutputFunc()(num_data_, score, cuda_score_convert_buffer_);
  LaunchEvalKernel(cuda_score_convert_buffer_);
  double sum_loss = 0.0f;
  CopyFromCUDADeviceToHostOuter<double>(&sum_loss, cuda_sum_loss_, 1, __FILE__, __LINE__);
  return std::vector<double>(1, sum_loss / static_cast<double>(num_data_));
}

CUDAKullbackLeiblerDivergence::CUDAKullbackLeiblerDivergence(const Config& config): KullbackLeiblerDivergence(config) {}

CUDAKullbackLeiblerDivergence::~CUDAKullbackLeiblerDivergence() {}

void CUDAKullbackLeiblerDivergence::Init(const Metadata& metadata, data_size_t num_data) {
  KullbackLeiblerDivergence::Init(metadata, num_data);
  cuda_label_ = metadata.cuda_metadata()->cuda_label();
  cuda_weights_ = metadata.cuda_metadata()->cuda_weights();
  AllocateCUDAMemoryOuter<double>(&cuda_score_convert_buffer_, static_cast<size_t>(num_data_), __FILE__, __LINE__);

  const data_size_t num_blocks = (num_data + EVAL_BLOCK_SIZE_XENTROPY_METRIC - 1) / EVAL_BLOCK_SIZE_XENTROPY_METRIC;
  AllocateCUDAMemoryOuter<double>(&cuda_sum_loss_buffer_, static_cast<size_t>(num_blocks), __FILE__, __LINE__);
  AllocateCUDAMemoryOuter<double>(&cuda_sum_loss_, 1, __FILE__, __LINE__);
}

std::vector<double> CUDAKullbackLeiblerDivergence::Eval(const double* score, const ObjectiveFunction* objective) const {
  objective->GetCUDAConvertOutputFunc()(num_data_, score, cuda_score_convert_buffer_);
  LaunchEvalKernel(cuda_score_convert_buffer_);
  double sum_loss = 0.0f;
  CopyFromCUDADeviceToHostOuter<double>(&sum_loss, cuda_sum_loss_, 1, __FILE__, __LINE__);
  return std::vector<double>(1, presum_label_entropy_ + sum_loss / sum_weights_);
}

}  // namespace LightGBM
