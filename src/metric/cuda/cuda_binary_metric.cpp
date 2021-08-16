/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include "cuda_binary_metric.hpp"

namespace LightGBM {

template <typename CUDAPointWiseLossCalculator>
CUDABinaryMetric<CUDAPointWiseLossCalculator>::CUDABinaryMetric(const Config& config): BinaryMetric<CUDAPointWiseLossCalculator>(config) {}

template <typename CUDAPointWiseLossCalculator>
CUDABinaryMetric<CUDAPointWiseLossCalculator>::~CUDABinaryMetric() {}

template <typename CUDAPointWiseLossCalculator>
void CUDABinaryMetric<CUDAPointWiseLossCalculator>::Init(const Metadata& metadata, data_size_t num_data) {
  BinaryMetric<CUDAPointWiseLossCalculator>::Init(metadata, num_data);

  cuda_label_ = metadata.cuda_metadata()->cuda_label();
  cuda_weights_ = metadata.cuda_metadata()->cuda_weights();

  const data_size_t num_blocks = (num_data + EVAL_BLOCK_SIZE_BINARY_METRIC - 1) / EVAL_BLOCK_SIZE_BINARY_METRIC;
  AllocateCUDAMemoryOuter<double>(&cuda_sum_loss_buffer_, static_cast<size_t>(num_blocks), __FILE__, __LINE__);
  AllocateCUDAMemoryOuter<double>(&cuda_score_convert_buffer_, static_cast<size_t>(num_data), __FILE__, __LINE__);
  AllocateCUDAMemoryOuter<double>(&cuda_sum_loss_, 1, __FILE__, __LINE__);
}

template <typename CUDAPointWiseLossCalculator>
std::vector<double> CUDABinaryMetric<CUDAPointWiseLossCalculator>::Eval(const double* score, const ObjectiveFunction* objective) const {
  double sum_loss = 0.0f;
  objective->GetCUDAConvertOutputFunc()(this->num_data_, score, cuda_score_convert_buffer_);
  LaunchEvalKernel(cuda_score_convert_buffer_);
  CopyFromCUDADeviceToHostOuter<double>(&sum_loss, cuda_sum_loss_, 1, __FILE__, __LINE__);
  return std::vector<double>(1, sum_loss / this->sum_weights_);
}

CUDABinaryLoglossMetric::CUDABinaryLoglossMetric(const Config& config): CUDABinaryMetric<CUDABinaryLoglossMetric>(config) {}

CUDABinaryErrorMetric::CUDABinaryErrorMetric(const Config& config) : CUDABinaryMetric<CUDABinaryErrorMetric>(config) {}

CUDAAUCMetric::CUDAAUCMetric(const Config& config): AUCMetric(config) {}

CUDAAUCMetric::~CUDAAUCMetric() {}

void CUDAAUCMetric::Init(const Metadata& metadata, data_size_t num_data) {
  AUCMetric::Init(metadata, num_data);
  AllocateCUDAMemoryOuter<data_size_t>(&cuda_indices_buffer_, static_cast<size_t>(num_data), __FILE__, __LINE__);
  AllocateCUDAMemoryOuter<double>(&cuda_sum_pos_buffer_, static_cast<size_t>(num_data), __FILE__, __LINE__);
  AllocateCUDAMemoryOuter<data_size_t>(&cuda_threshold_mark_, static_cast<size_t>(num_data), __FILE__, __LINE__);
  const data_size_t num_blocks = (num_data + EVAL_BLOCK_SIZE_BINARY_METRIC - 1) / EVAL_BLOCK_SIZE_BINARY_METRIC;
  AllocateCUDAMemoryOuter<double>(&cuda_block_sum_pos_buffer_, static_cast<size_t>(num_blocks) + 1, __FILE__, __LINE__);
  AllocateCUDAMemoryOuter<data_size_t>(&cuda_block_threshold_mark_buffer_, static_cast<size_t>(num_blocks), __FILE__, __LINE__);
  AllocateCUDAMemoryOuter<uint16_t>(&cuda_block_mark_first_zero_, static_cast<size_t>(num_blocks), __FILE__, __LINE__);
  cuda_weights_ = metadata.cuda_metadata()->cuda_weights();
  cuda_label_ = metadata.cuda_metadata()->cuda_label();
}

std::vector<double> CUDAAUCMetric::Eval(const double* score, const ObjectiveFunction*) const {
  LaunchEvalKernel(score);
}

}  // namespace LightGBM
