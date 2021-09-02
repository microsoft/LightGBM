/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_regression_objective.hpp"

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
  AllocateCUDAMemoryOuter<double>(&cuda_block_buffer_, static_cast<size_t>(num_get_gradients_blocks_), __FILE__, __LINE__);
  if (sqrt_) {
    InitCUDAMemoryFromHostMemoryOuter<label_t>(&cuda_trans_label_, trans_label_.data(), trans_label_.size(), __FILE__, __LINE__);
    cuda_labels_ = cuda_trans_label_;
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

void CUDARegressionL2loss::RenewTreeOutputCUDA(
  const double* score,
  const data_size_t* data_indices_in_leaf,
  const data_size_t* num_data_in_leaf,
  const data_size_t* data_start_in_leaf,
  const int num_leaves,
  double* leaf_value) const {
  global_timer.Start("CUDARegressionL1loss::LaunchRenewTreeOutputCUDAKernel");
  LaunchRenewTreeOutputCUDAKernel(score, data_indices_in_leaf, num_data_in_leaf, data_start_in_leaf, num_leaves, leaf_value);
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
  global_timer.Stop("CUDARegressionL1loss::LaunchRenewTreeOutputCUDAKernel");
}

CUDARegressionL1loss::CUDARegressionL1loss(const Config& config):
CUDARegressionL2loss(config) {}

CUDARegressionL1loss::CUDARegressionL1loss(const std::vector<std::string>& strs):
CUDARegressionL2loss(strs) {}

CUDARegressionL1loss::~CUDARegressionL1loss() {}

void CUDARegressionL1loss::Init(const Metadata& metadata, data_size_t num_data) {
  CUDARegressionL2loss::Init(metadata, num_data);
  AllocateCUDAMemoryOuter<data_size_t>(&cuda_data_indices_buffer_, static_cast<size_t>(num_data), __FILE__, __LINE__);
  AllocateCUDAMemoryOuter<label_t>(&cuda_percentile_result_, 1, __FILE__, __LINE__);
  if (cuda_weights_ != nullptr) {
    const int num_blocks = (num_data_ + GET_GRADIENTS_BLOCK_SIZE_REGRESSION - 1) / GET_GRADIENTS_BLOCK_SIZE_REGRESSION + 1;
    AllocateCUDAMemoryOuter<double>(&cuda_weights_prefix_sum_, static_cast<size_t>(num_data), __FILE__, __LINE__);
    AllocateCUDAMemoryOuter<double>(&cuda_weights_prefix_sum_buffer_, static_cast<size_t>(num_blocks), __FILE__, __LINE__);
    AllocateCUDAMemoryOuter<label_t>(&cuda_weight_by_leaf_buffer_, static_cast<size_t>(num_data_), __FILE__, __LINE__);
  }
  AllocateCUDAMemoryOuter<double>(&cuda_residual_buffer_, static_cast<size_t>(num_data_), __FILE__, __LINE__);
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

CUDARegressionPoissonLoss::CUDARegressionPoissonLoss(const Config& config):
CUDARegressionL2loss(config), max_delta_step_(config.poisson_max_delta_step) {
  if (sqrt_) {
    Log::Warning("Cannot use sqrt transform in %s Regression, will auto disable it", GetName());
    sqrt_ = false;
  }
}

CUDARegressionPoissonLoss::CUDARegressionPoissonLoss(const std::vector<std::string>& strs):
CUDARegressionL2loss(strs) {}

CUDARegressionPoissonLoss::~CUDARegressionPoissonLoss() {}

void CUDARegressionPoissonLoss::Init(const Metadata& metadata, data_size_t num_data) {
  CUDARegressionL2loss::Init(metadata, num_data);
  AllocateCUDAMemoryOuter<double>(&cuda_block_buffer_, static_cast<size_t>(num_get_gradients_blocks_), __FILE__, __LINE__);
  LaunchCheckLabelKernel();
}

double CUDARegressionPoissonLoss::LaunchCalcInitScoreKernel() const {
  return Common::SafeLog(CUDARegressionL2loss::LaunchCalcInitScoreKernel());
}

CUDARegressionQuantileloss::CUDARegressionQuantileloss(const Config& config):
CUDARegressionL2loss(config), alpha_(config.alpha) {
  CHECK(alpha_ > 0 && alpha_ < 1);
}

CUDARegressionQuantileloss::CUDARegressionQuantileloss(const std::vector<std::string>& strs):
CUDARegressionL2loss(strs) {}

void CUDARegressionQuantileloss::Init(const Metadata& metadata, data_size_t num_data) {
  CUDARegressionL2loss::Init(metadata, num_data);
  AllocateCUDAMemoryOuter<data_size_t>(&cuda_data_indices_buffer_, static_cast<size_t>(num_data), __FILE__, __LINE__);
  AllocateCUDAMemoryOuter<label_t>(&cuda_percentile_result_, 1, __FILE__, __LINE__);
  if (cuda_weights_ != nullptr) {
    const int num_blocks = (num_data_ + GET_GRADIENTS_BLOCK_SIZE_REGRESSION - 1) / GET_GRADIENTS_BLOCK_SIZE_REGRESSION + 1;
    AllocateCUDAMemoryOuter<double>(&cuda_weights_prefix_sum_, static_cast<size_t>(num_data), __FILE__, __LINE__);
    AllocateCUDAMemoryOuter<double>(&cuda_weights_prefix_sum_buffer_, static_cast<size_t>(num_blocks), __FILE__, __LINE__);
    AllocateCUDAMemoryOuter<label_t>(&cuda_weight_by_leaf_buffer_, static_cast<size_t>(num_data_), __FILE__, __LINE__);
  }
  AllocateCUDAMemoryOuter<double>(&cuda_residual_buffer_, static_cast<size_t>(num_data_), __FILE__, __LINE__);
}

CUDARegressionQuantileloss::~CUDARegressionQuantileloss() {}

CUDARegressionMAPELOSS::CUDARegressionMAPELOSS(const Config& config):
CUDARegressionL1loss(config) {}

CUDARegressionMAPELOSS::CUDARegressionMAPELOSS(const std::vector<std::string>& strs):
CUDARegressionL1loss(strs) {}

CUDARegressionMAPELOSS::~CUDARegressionMAPELOSS() {}

void CUDARegressionMAPELOSS::Init(const Metadata& metadata, data_size_t num_data) {
  CUDARegressionL1loss::Init(metadata, num_data);
  if (cuda_weights_ == nullptr) {
    // allocate buffer for weights when they are not allocated in L1 loss
    const int num_blocks = (num_data_ + GET_GRADIENTS_BLOCK_SIZE_REGRESSION - 1) / GET_GRADIENTS_BLOCK_SIZE_REGRESSION + 1;
    AllocateCUDAMemoryOuter<double>(&cuda_weights_prefix_sum_, static_cast<size_t>(num_data), __FILE__, __LINE__);
    AllocateCUDAMemoryOuter<double>(&cuda_weights_prefix_sum_buffer_, static_cast<size_t>(num_blocks), __FILE__, __LINE__);
    AllocateCUDAMemoryOuter<label_t>(&cuda_weight_by_leaf_buffer_, static_cast<size_t>(num_data_), __FILE__, __LINE__);
  }
  for (data_size_t i = 0; i < num_data_; ++i) {
    if (std::fabs(label_[i]) < 1) {
      Log::Warning(
        "Some label values are < 1 in absolute value. MAPE is unstable with such values, "
        "so LightGBM rounds them to 1.0 when calculating MAPE.");
      break;
    }
  }
  AllocateCUDAMemoryOuter<label_t>(&cuda_label_weights_, static_cast<size_t>(num_data_), __FILE__, __LINE__);
  LaunchCalcLabelWeightKernel();
}

CUDARegressionGammaLoss::CUDARegressionGammaLoss(const Config& config):
CUDARegressionPoissonLoss(config) {}

CUDARegressionGammaLoss::CUDARegressionGammaLoss(const std::vector<std::string>& strs):
CUDARegressionPoissonLoss(strs) {}

CUDARegressionGammaLoss::~CUDARegressionGammaLoss() {}

CUDARegressionTweedieLoss::CUDARegressionTweedieLoss(const Config& config):
CUDARegressionPoissonLoss(config), rho_(config.tweedie_variance_power) {}

CUDARegressionTweedieLoss::CUDARegressionTweedieLoss(const std::vector<std::string>& strs):
CUDARegressionPoissonLoss(strs) {}

CUDARegressionTweedieLoss::~CUDARegressionTweedieLoss() {}

}  // namespace LightGBM

#endif  // USE_CUDA
