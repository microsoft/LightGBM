/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA_EXP

#include "cuda_regression_objective.hpp"
#include <LightGBM/cuda/cuda_algorithms.hpp>

namespace LightGBM {

double CUDARegressionL2loss::LaunchCalcInitScoreKernel() const {
  double label_sum = 0.0f, weight_sum = 0.0f;
  if (cuda_weights_ == nullptr) {
    ShuffleReduceSumGlobal<label_t, double>(cuda_labels_, static_cast<size_t>(num_data_), cuda_block_buffer_);
    CopyFromCUDADeviceToHost<double>(&label_sum, cuda_block_buffer_, 1, __FILE__, __LINE__);
    weight_sum = static_cast<double>(num_data_);
  } else {
    ShuffleReduceDotProdGlobal<label_t, double>(cuda_labels_, cuda_weights_, static_cast<size_t>(num_data_), cuda_block_buffer_);
    CopyFromCUDADeviceToHost<double>(&label_sum, cuda_block_buffer_, 1, __FILE__, __LINE__);
    ShuffleReduceSumGlobal<label_t, double>(cuda_weights_, static_cast<size_t>(num_data_), cuda_block_buffer_);
    CopyFromCUDADeviceToHost<double>(&weight_sum, cuda_block_buffer_, 1, __FILE__, __LINE__);
  }
  return label_sum / weight_sum;
}

__global__ void ConvertOutputCUDAKernel_Regression(const bool sqrt, const data_size_t num_data, const double* input, double* output) {
  const int data_index = static_cast<data_size_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (data_index < num_data) {
    if (sqrt) {
      const double sign = input[data_index] >= 0.0f ? 1 : -1; 
      output[data_index] = sign * input[data_index] * input[data_index];
    } else {
      output[data_index] = input[data_index];
    }
  }
}

void CUDARegressionL2loss::LaunchConvertOutputCUDAKernel(const data_size_t num_data, const double* input, double* output) const {
  const int num_blocks = (num_data + GET_GRADIENTS_BLOCK_SIZE_REGRESSION - 1) / GET_GRADIENTS_BLOCK_SIZE_REGRESSION;
  ConvertOutputCUDAKernel_Regression<<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(sqrt_, num_data, input, output);
}

template <bool USE_WEIGHT>
__global__ void GetGradientsKernel_RegressionL2(const double* cuda_scores, const label_t* cuda_labels, const label_t* cuda_weights, const data_size_t num_data,
  score_t* cuda_out_gradients, score_t* cuda_out_hessians) {
  const data_size_t data_index = static_cast<data_size_t>(blockDim.x * blockIdx.x + threadIdx.x);
  if (data_index < num_data) {
    if (!USE_WEIGHT) {
      cuda_out_gradients[data_index] = static_cast<score_t>(cuda_scores[data_index] - cuda_labels[data_index]);
      cuda_out_hessians[data_index] = 1.0f;
    } else {
      const score_t weight = static_cast<score_t>(cuda_weights[data_index]);
      cuda_out_gradients[data_index] = static_cast<score_t>(cuda_scores[data_index] - cuda_labels[data_index]) * weight;
      cuda_out_hessians[data_index] = weight;
    }
  }
}

void CUDARegressionL2loss::LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const {
  const int num_blocks = (num_data_ + GET_GRADIENTS_BLOCK_SIZE_REGRESSION - 1) / GET_GRADIENTS_BLOCK_SIZE_REGRESSION;
  if (cuda_weights_ == nullptr) {
    GetGradientsKernel_RegressionL2<false><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(score, cuda_labels_, nullptr, num_data_, gradients, hessians);
  } else {
    GetGradientsKernel_RegressionL2<true><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(score, cuda_labels_, cuda_weights_, num_data_, gradients, hessians);
  }
}


}  // namespace LightGBM

#endif  // USE_CUDA_EXP
