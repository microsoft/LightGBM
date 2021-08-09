/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include "cuda_multiclass_objective.hpp"

namespace LightGBM {

CUDAMulticlassSoftmax::CUDAMulticlassSoftmax(const Config& config): MulticlassSoftmax(config) {}

CUDAMulticlassSoftmax::CUDAMulticlassSoftmax(const std::vector<std::string>& strs): MulticlassSoftmax(strs) {}

CUDAMulticlassSoftmax::~CUDAMulticlassSoftmax() {}

void CUDAMulticlassSoftmax::Init(const Metadata& metadata, data_size_t num_data) {
  MulticlassSoftmax::Init(metadata, num_data);
  cuda_label_ = metadata.cuda_metadata()->cuda_label();
  cuda_weights_ = metadata.cuda_metadata()->cuda_weights();
  AllocateCUDAMemoryOuter<double>(&cuda_boost_from_score_, num_class_, __FILE__, __LINE__);
  AllocateCUDAMemoryOuter<double>(&cuda_softmax_buffer_, static_cast<size_t>(num_data) * static_cast<size_t>(num_class_), __FILE__, __LINE__);
  SetCUDAMemoryOuter<double>(cuda_boost_from_score_, 0, num_class_, __FILE__, __LINE__);
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
}

void CUDAMulticlassSoftmax::GetGradients(const double* score, score_t* gradients, score_t* hessians) const {
  LaunchGetGradientsKernel(score, gradients, hessians);
  /*std::vector<score_t> cpu_gradients(100, 0.0f);
  std::vector<score_t> cpu_hessians(100, 0.0f);
  CopyFromCUDADeviceToHostOuter<score_t>(cpu_gradients.data(), gradients, 100, __FILE__, __LINE__);
  CopyFromCUDADeviceToHostOuter<score_t>(cpu_hessians.data(), hessians, 100, __FILE__, __LINE__);
  for (size_t i = 0; i < 100; ++i) {
    Log::Warning("class 0 data %d gradient = %f, hessians = %f", i, cpu_gradients[i], cpu_hessians[i]);
  }
  CopyFromCUDADeviceToHostOuter<score_t>(cpu_gradients.data(), gradients + num_data_ - 100, 100, __FILE__, __LINE__);
  CopyFromCUDADeviceToHostOuter<score_t>(cpu_hessians.data(), hessians + num_data_ - 100, 100, __FILE__, __LINE__);
  for (size_t i = 0; i < 100; ++i) {
    Log::Warning("class 0 data %d gradient = %f, hessians = %f", i + num_data_ - 100, cpu_gradients[i], cpu_hessians[i]);
  }
  CopyFromCUDADeviceToHostOuter<score_t>(cpu_gradients.data(), gradients + num_data_, 100, __FILE__, __LINE__);
  CopyFromCUDADeviceToHostOuter<score_t>(cpu_hessians.data(), hessians + num_data_, 100, __FILE__, __LINE__);
  for (size_t i = 0; i < 100; ++i) {
    Log::Warning("class 1 data %d gradient = %f, hessians = %f", i, cpu_gradients[i], cpu_hessians[i]);
  }*/
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
}

void CUDAMulticlassSoftmax::ConvertOutputCUDA(const data_size_t num_data, const double* input, double* output) const {
  LaunchConvertOutputCUDAKernel(num_data, input, output);
}

CUDAMulticlassOVA::CUDAMulticlassOVA(const Config& config) {
  num_class_ = config.num_class;
  for (int i = 0; i < num_class_; ++i) {
    binary_loss_.emplace_back(new CUDABinaryLogloss(config, i));
  }
  sigmoid_ = config.sigmoid;
}

CUDAMulticlassOVA::CUDAMulticlassOVA(const std::vector<std::string>& strs): MulticlassOVA(strs) {}

CUDAMulticlassOVA::~CUDAMulticlassOVA() {}

}  // namespace LightGBM
