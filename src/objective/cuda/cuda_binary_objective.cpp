/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_binary_objective.hpp"

namespace LightGBM {

CUDABinaryLogloss::CUDABinaryLogloss(const Config& config,
                  std::function<bool(label_t)> is_pos):
BinaryLogloss(config, is_pos) {}

CUDABinaryLogloss::CUDABinaryLogloss(const std::vector<std::string>& strs): BinaryLogloss(strs) {}

CUDABinaryLogloss::~CUDABinaryLogloss() {}

void CUDABinaryLogloss::Init(const Metadata& metadata, data_size_t num_data) {
  BinaryLogloss::Init(metadata, num_data);
  cuda_label_ = metadata.cuda_metadata()->cuda_label();
  cuda_weights_ = metadata.cuda_metadata()->cuda_weights();
  AllocateCUDAMemoryOuter<double>(&cuda_boost_from_score_, 1, __FILE__, __LINE__);
  SetCUDAMemoryOuter<double>(cuda_boost_from_score_, 0, 1, __FILE__, __LINE__);
}

void CUDABinaryLogloss::GetGradients(const double* scores, score_t* gradients, score_t* hessians) const {
  LaunchGetGradientsKernel(scores, gradients, hessians);
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
}

double CUDABinaryLogloss::BoostFromScore(int) const {
  LaunchBoostFromScoreKernel();
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
  double boost_from_score = 0.0f;
  CopyFromCUDADeviceToHostOuter<double>(&boost_from_score, cuda_boost_from_score_, 1, __FILE__, __LINE__);
  return boost_from_score;
}

}  // namespace LightGBM

#endif  // USE_CUDA
