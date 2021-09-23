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
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
  for (int class_index = 0; class_index < num_class_; ++class_index) {
    std::vector<score_t> host_gradients(num_data_, 0.0f);
    std::vector<score_t> host_hessians(num_data_, 0.0f);
    const size_t offset = static_cast<size_t>(class_index * num_data_);
    CopyFromCUDADeviceToHostOuter<score_t>(host_gradients.data(), gradients + offset, static_cast<size_t>(num_data_), __FILE__, __LINE__);
    CopyFromCUDADeviceToHostOuter<score_t>(host_hessians.data(), hessians + offset, static_cast<size_t>(num_data_), __FILE__, __LINE__);
    const int num_threads = OMP_NUM_THREADS();
    std::vector<score_t> thread_abs_max_gradient(num_threads, 0.0f);
    std::vector<score_t> thread_abs_max_hessian(num_threads, 0.0f);
    std::vector<score_t> thread_abs_min_hessian(num_threads, std::numeric_limits<score_t>::infinity());
    Threading::For<data_size_t>(0, num_data_, 512,
      [&thread_abs_max_gradient, &thread_abs_max_hessian, &thread_abs_min_hessian, &host_gradients, &host_hessians] (int thread_index, data_size_t start, data_size_t end) {
        for (data_size_t index = start; index < end; ++index) {
          const score_t gradient = host_gradients[index];
          const score_t hessian = host_hessians[index];
          if (std::fabs(gradient) > std::fabs(thread_abs_max_gradient[thread_index])) {
            thread_abs_max_gradient[thread_index] = gradient;
          }
          if (std::fabs(hessian) > std::fabs(thread_abs_max_hessian[thread_index])) {
            thread_abs_max_hessian[thread_index] = hessian;
          }
          if (std::fabs(hessian) < std::fabs(thread_abs_min_hessian[thread_index])) {
            thread_abs_min_hessian[thread_index] = hessian;
          }
        }
      });
    double max_abs_gradient = 0.0f;
    double max_abs_hessian = 0.0f;
    double min_abs_hessian = std::numeric_limits<score_t>::infinity();
    for (int thread_index = 0; thread_index < num_threads; ++thread_index) {
      if (std::fabs(thread_abs_max_gradient[thread_index]) > std::fabs(max_abs_gradient)) {
        max_abs_gradient = thread_abs_max_gradient[thread_index];
      }
      if (std::fabs(thread_abs_max_hessian[thread_index] > std::fabs(max_abs_hessian))) {
        max_abs_hessian = thread_abs_max_hessian[thread_index];
      }
      if (std::fabs(thread_abs_min_hessian[thread_index] < std::fabs(min_abs_hessian))) {
        min_abs_hessian = thread_abs_min_hessian[thread_index];
      }
    }
    Log::Warning("class %d max_abs_gradient = %f, max_abs_hessian = %f, min_abs_hessian = %f", class_index, max_abs_gradient, max_abs_hessian, min_abs_hessian);
  }
}

void CUDAMulticlassSoftmax::ConvertOutputCUDA(const data_size_t num_data, const double* input, double* output) const {
  LaunchConvertOutputCUDAKernel(num_data, input, output);
}

CUDAMulticlassOVA::CUDAMulticlassOVA(const Config& config) {
  num_class_ = config.num_class;
  for (int i = 0; i < num_class_; ++i) {
    cuda_binary_loss_.emplace_back(new CUDABinaryLogloss(config, i));
  }
  sigmoid_ = config.sigmoid;
}

CUDAMulticlassOVA::CUDAMulticlassOVA(const std::vector<std::string>& strs): MulticlassOVA(strs) {}

CUDAMulticlassOVA::~CUDAMulticlassOVA() {}

void CUDAMulticlassOVA::Init(const Metadata& metadata, data_size_t num_data) {
  num_data_ = num_data;
  for (int i = 0; i < num_class_; ++i) {
    cuda_binary_loss_[i]->Init(metadata, num_data);
  }
}

void CUDAMulticlassOVA::GetGradients(const double* score, score_t* gradients, score_t* hessians) const {
  for (int i = 0; i < num_class_; ++i) {
    int64_t offset = static_cast<int64_t>(num_data_) * i;
    cuda_binary_loss_[i]->GetGradients(score + offset, gradients + offset, hessians + offset);
  }
}

void CUDAMulticlassOVA::ConvertOutputCUDA(const data_size_t num_data, const double* input, double* output) const {
  for (int i = 0; i < num_class_; ++i) {
    cuda_binary_loss_[i]->ConvertOutputCUDA(num_data, input + i * num_data, output + i * num_data);
  }
}

}  // namespace LightGBM
