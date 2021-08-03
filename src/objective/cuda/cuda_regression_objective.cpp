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
  AllocateCUDAMemoryOuter<double>(&cuda_boost_from_score_, 1, __FILE__, __LINE__);
  SetCUDAMemoryOuter<double>(cuda_boost_from_score_, 0, 1, __FILE__, __LINE__);
}

void CUDARegressionL2loss::GetGradients(const double* score, score_t* gradients, score_t* hessians) const {
  LaunchGetGradientsKernel(score, gradients, hessians);
}

double CUDARegressionL2loss::BoostFromScore(int) const {
  LaunchCalcInitScoreKernel();
  double boost_from_score = 0.0f;
  CopyFromCUDADeviceToHostOuter<double>(&boost_from_score, cuda_boost_from_score_, 1, __FILE__, __LINE__);
  return boost_from_score;
}

}  // namespace LightGBM

#endif  // USE_CUDA
