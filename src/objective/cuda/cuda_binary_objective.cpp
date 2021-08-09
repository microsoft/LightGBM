/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_binary_objective.hpp"

namespace LightGBM {

CUDABinaryLogloss::CUDABinaryLogloss(const Config& config, const int ova_class_id):
BinaryLogloss(config), ova_class_id_(ova_class_id) {}

CUDABinaryLogloss::CUDABinaryLogloss(const std::vector<std::string>& strs): BinaryLogloss(strs) {}

CUDABinaryLogloss::~CUDABinaryLogloss() {}

void CUDABinaryLogloss::Init(const Metadata& metadata, data_size_t num_data) {
  BinaryLogloss::Init(metadata, num_data);
  cuda_label_ = metadata.cuda_metadata()->cuda_label();
  cuda_weights_ = metadata.cuda_metadata()->cuda_weights();
  AllocateCUDAMemoryOuter<double>(&cuda_boost_from_score_, 1, __FILE__, __LINE__);
  SetCUDAMemoryOuter<double>(cuda_boost_from_score_, 0, 1, __FILE__, __LINE__);
  if (label_weights_[0] != 1.0f || label_weights_[1] != 1.0f) {
    InitCUDAMemoryFromHostMemoryOuter<double>(&cuda_label_weights_, label_weights_, 2, __FILE__, __LINE__);
  } else {
    cuda_label_weights_ = nullptr;
  }
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

void CUDABinaryLogloss::ConvertOutputCUDA(const data_size_t num_data, const double* input, double* output) const {
  LaunchConvertOutputCUDAKernel(num_data, input, output);
}

}  // namespace LightGBM

#endif  // USE_CUDA
