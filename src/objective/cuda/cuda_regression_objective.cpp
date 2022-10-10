/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA_EXP

#include "cuda_regression_objective.hpp"

#include <string>
#include <vector>

namespace LightGBM {

CUDARegressionL2loss::CUDARegressionL2loss(const Config& config):
RegressionL2loss(config) {}

CUDARegressionL2loss::CUDARegressionL2loss(const std::vector<std::string>& strs):
RegressionL2loss(strs) {}

CUDARegressionL2loss::~CUDARegressionL2loss() {}

void CUDARegressionL2loss::Init(const Metadata& metadata, data_size_t num_data) {
  RegressionL2loss::Init(metadata, num_data);
  cuda_labels_ = metadata.cuda_metadata()->cuda_label();
  cuda_weights_ = metadata.cuda_metadata()->cuda_weights();
  num_get_gradients_blocks_ = (num_data_ + GET_GRADIENTS_BLOCK_SIZE_REGRESSION - 1) / GET_GRADIENTS_BLOCK_SIZE_REGRESSION;
  cuda_block_buffer_.Resize(static_cast<size_t>(num_get_gradients_blocks_));
  if (sqrt_) {
    cuda_trans_label_.Resize(trans_label_.size());
    CopyFromHostToCUDADevice<label_t>(cuda_trans_label_.RawData(), trans_label_.data(), trans_label_.size(), __FILE__, __LINE__);
    cuda_labels_ = cuda_trans_label_.RawData();
  }
}

void CUDARegressionL2loss::GetGradients(const double* score, score_t* gradients, score_t* hessians) const {
  LaunchGetGradientsKernel(score, gradients, hessians);
}

double CUDARegressionL2loss::BoostFromScore(int) const {
  return LaunchCalcInitScoreKernel();
}

void CUDARegressionL2loss::ConvertOutputCUDA(const data_size_t num_data, const double* input, double* output) const {
  LaunchConvertOutputCUDAKernel(num_data, input, output);
}


CUDARegressionL1loss::CUDARegressionL1loss(const Config& config):
CUDARegressionL2loss(config) {}

CUDARegressionL1loss::CUDARegressionL1loss(const std::vector<std::string>& strs):
CUDARegressionL2loss(strs) {}

CUDARegressionL1loss::~CUDARegressionL1loss() {}

void CUDARegressionL1loss::Init(const Metadata& metadata, data_size_t num_data) {
  CUDARegressionL2loss::Init(metadata, num_data);
  cuda_data_indices_buffer_.Resize(static_cast<size_t>(num_data));
  cuda_percentile_result_.Resize(1);
  if (cuda_weights_ != nullptr) {
    const int num_blocks = (num_data + GET_GRADIENTS_BLOCK_SIZE_REGRESSION - 1) / GET_GRADIENTS_BLOCK_SIZE_REGRESSION + 1;
    cuda_weights_prefix_sum_.Resize(static_cast<size_t>(num_data));
    cuda_weights_prefix_sum_buffer_.Resize(static_cast<size_t>(num_blocks));
    cuda_weight_by_leaf_buffer_.Resize(static_cast<size_t>(num_data));
  }
  cuda_residual_buffer_.Resize(static_cast<size_t>(num_data));
}

void CUDARegressionL1loss::RenewTreeOutputCUDA(
  const double* score,
  const data_size_t* data_indices_in_leaf,
  const data_size_t* num_data_in_leaf,
  const data_size_t* data_start_in_leaf,
  const int num_leaves,
  double* leaf_value) const {
  global_timer.Start("CUDARegressionL1loss::LaunchRenewTreeOutputCUDAKernel");
  LaunchRenewTreeOutputCUDAKernel(score, data_indices_in_leaf, num_data_in_leaf, data_start_in_leaf, num_leaves, leaf_value);
  SynchronizeCUDADevice(__FILE__, __LINE__);
  global_timer.Stop("CUDARegressionL1loss::LaunchRenewTreeOutputCUDAKernel");
}


CUDARegressionHuberLoss::CUDARegressionHuberLoss(const Config& config):
CUDARegressionL2loss(config), alpha_(config.alpha) {
  if (sqrt_) {
    Log::Warning("Cannot use sqrt transform in %s Regression, will auto disable it", GetName());
    sqrt_ = false;
  }
}

CUDARegressionHuberLoss::CUDARegressionHuberLoss(const std::vector<std::string>& strs):
CUDARegressionL2loss(strs) {}

CUDARegressionHuberLoss::~CUDARegressionHuberLoss() {}


CUDARegressionFairLoss::CUDARegressionFairLoss(const Config& config):
CUDARegressionL2loss(config), c_(config.fair_c) {}

CUDARegressionFairLoss::CUDARegressionFairLoss(const std::vector<std::string>& strs):
CUDARegressionL2loss(strs) {}

CUDARegressionFairLoss::~CUDARegressionFairLoss() {}


}  // namespace LightGBM

#endif  // USE_CUDA_EXP
