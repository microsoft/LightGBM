/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <LightGBM/cuda/cuda_algorithms.hpp>
#include "cuda_xentropy_objective.hpp"

namespace LightGBM {

double CUDACrossEntropy::LaunchCalcInitScoreKernel() const {
  double suml = 0.0f;
  double sumw = 0.0f;
  if (cuda_weights_ == nullptr) {
    sumw = static_cast<double>(num_data_);
    ReduceSumGlobal<label_t, double>(cuda_labels_, static_cast<size_t>(num_data_), cuda_reduce_sum_buffer_);
    CopyFromCUDADeviceToHostOuter<double>(&suml, cuda_reduce_sum_buffer_, 1, __FILE__, __LINE__);
  } else {
    ReduceDotProductGlobal<label_t, label_t, double>(cuda_labels_, cuda_weights_, static_cast<size_t>(num_data_), cuda_reduce_sum_buffer_);
    CopyFromCUDADeviceToHostOuter<double>(&suml, cuda_reduce_sum_buffer_, 1, __FILE__, __LINE__);
    ReduceSumGlobal<label_t, double>(cuda_weights_, static_cast<size_t>(num_data_), cuda_reduce_sum_buffer_);
    CopyFromCUDADeviceToHostOuter<double>(&sumw, cuda_reduce_sum_buffer_, 1, __FILE__, __LINE__);
  }
  double pavg = suml / sumw;
  pavg = std::min(pavg, 1.0 - kEpsilon);
  pavg = std::max<double>(pavg, kEpsilon);
  return std::log(pavg / (1.0f - pavg));
}

template <bool USE_WEIGHT>
__global__ void GetGradientsKernel_CrossEntropy(
  const double* cuda_scores,
  const label_t* cuda_labels,
  const label_t* cuda_weights,
  const data_size_t num_data,
  score_t* cuda_out_gradients,
  score_t* cuda_out_hessians) {
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (data_index < num_data) {
    if (USE_WEIGHT) {
      const double z = 1.0f / (1.0f + exp(-cuda_scores[data_index]));
      const label_t weight = cuda_weights[data_index];
      cuda_out_gradients[data_index] = static_cast<score_t>(z - cuda_labels[data_index] * weight);
      cuda_out_hessians[data_index] = static_cast<score_t>(z * (1.0f - z) * weight);
    } else {
      const double z = 1.0f / (1.0f + exp(-cuda_scores[data_index]));
      cuda_out_gradients[data_index] = static_cast<score_t>(z - cuda_labels[data_index]);
      cuda_out_hessians[data_index] = static_cast<score_t>(z * (1.0f - z));
    }
  }
}

void CUDACrossEntropy::LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const {
  const int num_blocks = (num_data_ + GET_GRADIENTS_BLOCK_SIZE_XENTROPY - 1) / GET_GRADIENTS_BLOCK_SIZE_XENTROPY;
  if (cuda_weights_ == nullptr) {
    GetGradientsKernel_CrossEntropy<false><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_XENTROPY>>>(score, cuda_labels_, nullptr, num_data_, gradients, hessians);
  } else {
    GetGradientsKernel_CrossEntropy<true><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_XENTROPY>>>(score, cuda_labels_, cuda_weights_, num_data_, gradients, hessians);
  }
}

double CUDACrossEntropyLambda::LaunchCalcInitScoreKernel() const {
  double suml = 0.0f;
  double sumw = 0.0f;
  if (cuda_weights_ == nullptr) {
    sumw = static_cast<double>(num_data_);
    ReduceSumGlobal<label_t, double>(cuda_labels_, static_cast<size_t>(num_data_), cuda_reduce_sum_buffer_);
    CopyFromCUDADeviceToHostOuter<double>(&suml, cuda_reduce_sum_buffer_, 1, __FILE__, __LINE__);
  } else {
    ReduceDotProductGlobal<label_t, label_t, double>(cuda_labels_, cuda_weights_, static_cast<size_t>(num_data_), cuda_reduce_sum_buffer_);
    CopyFromCUDADeviceToHostOuter<double>(&suml, cuda_reduce_sum_buffer_, 1, __FILE__, __LINE__);
    ReduceSumGlobal<label_t, double>(cuda_weights_, static_cast<size_t>(num_data_), cuda_reduce_sum_buffer_);
    CopyFromCUDADeviceToHostOuter<double>(&sumw, cuda_reduce_sum_buffer_, 1, __FILE__, __LINE__);
  }
  double havg = suml / sumw;
  return std::log(std::exp(havg) - 1.0f);
}

template <bool USE_WEIGHT>
__global__ void GetGradientsKernel_CrossEntropyLambda(
  const double* cuda_scores,
  const label_t* cuda_labels,
  const label_t* cuda_weights,
  const data_size_t num_data,
  score_t* cuda_out_gradients,
  score_t* cuda_out_hessians) {
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (data_index < num_data) {
    if (USE_WEIGHT) {
      const double w = static_cast<double>(cuda_weights[data_index]);
      const double y = static_cast<double>(cuda_labels[data_index]);
      const double epf = exp(cuda_scores[data_index]);
      const double hhat = log(1.0f + epf);
      const double z = 1.0f - exp(-w * hhat);
      const double enf = 1.0f / epf;  // = std::exp(-cuda_scores[data_index]);
      cuda_out_gradients[data_index] = static_cast<score_t>((1.0f - y / z) * w / (1.0f + enf));
      const double c = 1.0f / (1.0f - z);
      double d = 1.0f + epf;
      const double a = w * epf / (d * d);
      d = c - 1.0f;
      const double b = (c / (d * d) ) * (1.0f + w * epf - c);
      cuda_out_hessians[data_index] = static_cast<score_t>(a * (1.0f + y * b));
    } else {
      const double z = 1.0f / (1.0f + exp(-cuda_scores[data_index]));
      cuda_out_gradients[data_index] = static_cast<score_t>(z - cuda_labels[data_index]);
      cuda_out_hessians[data_index] = static_cast<score_t>(z * (1.0f - z));
    }
  }
}

void CUDACrossEntropyLambda::LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const {
  const int num_blocks = (num_data_ + GET_GRADIENTS_BLOCK_SIZE_XENTROPY - 1) / GET_GRADIENTS_BLOCK_SIZE_XENTROPY;
  if (cuda_weights_ == nullptr) {
    GetGradientsKernel_CrossEntropyLambda<false><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_XENTROPY>>>(score, cuda_labels_, nullptr, num_data_, gradients, hessians);
  } else {
    GetGradientsKernel_CrossEntropyLambda<true><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_XENTROPY>>>(score, cuda_labels_, cuda_weights_, num_data_, gradients, hessians);
  }
}

}  // namespace LightGBM
