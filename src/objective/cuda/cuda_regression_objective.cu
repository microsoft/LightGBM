/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_regression_objective.hpp"

namespace LightGBM {

__global__ void CalcInitScoreKernel_1_Regression(const label_t* cuda_labels, const data_size_t num_data, double* out_cuda_boost_from_score) {
  __shared__ label_t shared_label[CALC_INIT_SCORE_BLOCK_SIZE_REGRESSION];
  const unsigned int tid = threadIdx.x;
  const unsigned int i = (blockIdx.x * blockDim.x + tid) * NUM_DATA_THREAD_ADD_CALC_INIT_SCORE_REGRESSION;
  shared_label[tid] = 0.0f;
  __syncthreads();
  for (unsigned int j = 0; j < NUM_DATA_THREAD_ADD_CALC_INIT_SCORE_REGRESSION; ++j) {
    if (i + j < num_data) {
      shared_label[tid] += cuda_labels[i + j];
    }
  }
  __syncthreads();
  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0 && (tid + s) < CALC_INIT_SCORE_BLOCK_SIZE_REGRESSION) {
      shared_label[tid] += shared_label[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    atomicAdd_system(out_cuda_boost_from_score, shared_label[0]);
  }
}

__global__ void CalcInitScoreKernel_2_Regression(double* out_cuda_boost_from_score, const data_size_t num_data) {
  const double suml = *out_cuda_boost_from_score;
  const double sumw = static_cast<double>(num_data);
  const double init_score = suml / sumw;
  *out_cuda_boost_from_score = init_score;
}

void CUDARegressionL2loss::LaunchCalcInitScoreKernel() const {
  const data_size_t num_data_per_block = CALC_INIT_SCORE_BLOCK_SIZE_REGRESSION * NUM_DATA_THREAD_ADD_CALC_INIT_SCORE_REGRESSION;
  const int num_blocks = (num_data_ + num_data_per_block - 1) / num_data_per_block;
  CalcInitScoreKernel_1_Regression<<<num_blocks, CALC_INIT_SCORE_BLOCK_SIZE_REGRESSION>>>(cuda_labels_, num_data_, cuda_boost_from_score_);
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
  CalcInitScoreKernel_2_Regression<<<1, 1>>>(cuda_boost_from_score_, num_data_);
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
}

__global__ void GetGradientsKernel_Regression(const double* cuda_scores, const label_t* cuda_labels, const data_size_t num_data,
  score_t* cuda_out_gradients, score_t* cuda_out_hessians) {
  const data_size_t data_index = static_cast<data_size_t>(blockDim.x * blockIdx.x + threadIdx.x);
  if (data_index < num_data) {
    cuda_out_gradients[data_index] = static_cast<score_t>(cuda_scores[data_index] - cuda_labels[data_index]);
    cuda_out_hessians[data_index] = 1.0f;
  }
}

void CUDARegressionL2loss::LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const {
  const int num_blocks = (num_data_ + GET_GRADIENTS_BLOCK_SIZE_REGRESSION - 1) / GET_GRADIENTS_BLOCK_SIZE_REGRESSION;
  GetGradientsKernel_Regression<<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(score, cuda_labels_, num_data_, gradients, hessians);
}

}  // namespace LightGBM

#endif  // USE_CUDA
