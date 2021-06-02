/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_binary_objective.hpp"

namespace LightGBM {

__global__ void CalcInitScoreKernel_1(const label_t* cuda_labels, const data_size_t num_data, double* out_cuda_init_score) {
  __shared__ label_t shared_label[CALC_INIT_SCORE_BLOCK_SIZE];
  const unsigned int tid = threadIdx.x;
  const unsigned int i = (blockIdx.x * blockDim.x + tid) * NUM_DATA_THREAD_ADD_CALC_INIT_SCORE;
  shared_label[tid] = 0.0f;
  __syncthreads();
  for (unsigned int j = 0; j < NUM_DATA_THREAD_ADD_CALC_INIT_SCORE; ++j) {
    if (i + j < num_data) {
      shared_label[tid] += cuda_labels[i + j];
    }
  }
  __syncthreads();
  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0 && (tid + s) < CALC_INIT_SCORE_BLOCK_SIZE) {
      shared_label[tid] += shared_label[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    atomicAdd_system(out_cuda_init_score, shared_label[0]);
  }
}

__global__ void CalcInitScoreKernel_2(double* out_cuda_init_score, const data_size_t num_data, const double sigmoid) {
  const double suml = *out_cuda_init_score;
  const double sumw = static_cast<double>(num_data);
  const double pavg = suml / sumw;
  const double init_score = log(pavg / (1.0f - pavg)) / sigmoid;
  *out_cuda_init_score = init_score;
}

void CUDABinaryObjective::LaunchCalcInitScoreKernel() {
  const data_size_t num_data_per_block = CALC_INIT_SCORE_BLOCK_SIZE * NUM_DATA_THREAD_ADD_CALC_INIT_SCORE;
  const int num_blocks = (num_data_ + num_data_per_block - 1) / num_data_per_block;
  CalcInitScoreKernel_1<<<num_blocks, CALC_INIT_SCORE_BLOCK_SIZE>>>(cuda_labels_, num_data_, cuda_init_score_);
  SynchronizeCUDADevice();
  CalcInitScoreKernel_2<<<1, 1>>>(cuda_init_score_, num_data_, sigmoid_);
  SynchronizeCUDADevice();
}

__global__ void GetGradientsKernel(const double* cuda_scores, const label_t* cuda_labels,
  const double sigmoid, const data_size_t num_data,
  score_t* cuda_out_gradients, score_t* cuda_out_hessians) {
  const data_size_t data_index = static_cast<data_size_t>(blockDim.x * blockIdx.x + threadIdx.x);
  if (data_index < num_data) {
    const label_t cuda_label = static_cast<int>(cuda_labels[data_index]);
    const int label = cuda_label == 0 ? -1 : 1;
    const double response = -label * sigmoid / (1.0f + std::exp(label * sigmoid * cuda_scores[data_index]));
    const double abs_response = fabs(response);
    cuda_out_gradients[data_index] = static_cast<score_t>(response);
    cuda_out_hessians[data_index] = static_cast<score_t>(abs_response * (sigmoid - abs_response));
  }
}

void CUDABinaryObjective::LaunchGetGradientsKernel(const double* cuda_scores, score_t* cuda_out_gradients, score_t* cuda_out_hessians) {
  const int num_blocks = (num_data_ + GET_GRADIENTS_BLOCK_SIZE - 1) / GET_GRADIENTS_BLOCK_SIZE;
  GetGradientsKernel<<<num_blocks, GET_GRADIENTS_BLOCK_SIZE>>>(cuda_scores, cuda_labels_, sigmoid_, num_data_,
    cuda_out_gradients, cuda_out_hessians);
}

}  // namespace LightGBM

#endif  // USE_CUDA
