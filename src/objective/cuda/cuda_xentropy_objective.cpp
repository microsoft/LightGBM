/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include "cuda_xentropy_objective.hpp"

namespace LightGBM {

CUDACrossEntropy::CUDACrossEntropy(const Config& config): CrossEntropy(config) {}

CUDACrossEntropy::CUDACrossEntropy(const std::vector<std::string>& strs): CrossEntropy(strs) {}

CUDACrossEntropy::~CUDACrossEntropy() {}

void CUDACrossEntropy::Init(const Metadata& metadata, data_size_t num_data) {
  CrossEntropy::Init(metadata, num_data);
  const int num_blocks = (num_data + GET_GRADIENTS_BLOCK_SIZE_XENTROPY - 1) / GET_GRADIENTS_BLOCK_SIZE_XENTROPY;
  AllocateCUDAMemoryOuter<double>(&cuda_reduce_sum_buffer_, static_cast<size_t>(num_blocks), __FILE__, __LINE__);
  cuda_labels_ = metadata.cuda_metadata()->cuda_label();
  cuda_weights_ = metadata.cuda_metadata()->cuda_weights();
}

double CUDACrossEntropy::BoostFromScore(int) const {
  return LaunchCalcInitScoreKernel();
}

void CUDACrossEntropy::GetGradients(const double* score, score_t* gradients, score_t* hessians) const {
  LaunchGetGradientsKernel(score, gradients, hessians);
}

void CUDACrossEntropy::ConvertOutputCUDA(const data_size_t num_data, const double* input, double* output) const {
  LaunchConvertOutputCUDAKernel(num_data, input, output);
}

CUDACrossEntropyLambda::CUDACrossEntropyLambda(const Config& config): CrossEntropyLambda(config) {}

CUDACrossEntropyLambda::CUDACrossEntropyLambda(const std::vector<std::string>& strs): CrossEntropyLambda(strs) {}

CUDACrossEntropyLambda::~CUDACrossEntropyLambda() {}

void CUDACrossEntropyLambda::Init(const Metadata& metadata, data_size_t num_data) {
  CrossEntropyLambda::Init(metadata, num_data);
  const int num_blocks = (num_data + GET_GRADIENTS_BLOCK_SIZE_XENTROPY - 1) / GET_GRADIENTS_BLOCK_SIZE_XENTROPY;
  AllocateCUDAMemoryOuter<double>(&cuda_reduce_sum_buffer_, static_cast<size_t>(num_blocks), __FILE__, __LINE__);
  cuda_labels_ = metadata.cuda_metadata()->cuda_label();
  cuda_weights_ = metadata.cuda_metadata()->cuda_weights();
}

double CUDACrossEntropyLambda::BoostFromScore(int) const {
  return LaunchCalcInitScoreKernel();
}

void CUDACrossEntropyLambda::GetGradients(const double* score, score_t* gradients, score_t* hessians) const {
  LaunchGetGradientsKernel(score, gradients, hessians);
}

void CUDACrossEntropyLambda::ConvertOutputCUDA(const data_size_t num_data, const double* input, double* output) const {
  LaunchConvertOutputCUDAKernel(num_data, input, output);
}

}  // namespace LightGBM
