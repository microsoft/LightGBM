/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_binary_objective.hpp"

namespace LightGBM {

__global__ void BoostFromScoreKernel_1_BinaryLogloss(const label_t* cuda_labels, const data_size_t num_data, double* out_cuda_init_score) {
  __shared__ label_t shared_label[CALC_INIT_SCORE_BLOCK_SIZE_BINARY];
  const unsigned int tid = threadIdx.x;
  const unsigned int i = (blockIdx.x * blockDim.x + tid) * NUM_DATA_THREAD_ADD_CALC_INIT_SCORE_BINARY;
  shared_label[tid] = 0.0f;
  __syncthreads();
  for (unsigned int j = 0; j < NUM_DATA_THREAD_ADD_CALC_INIT_SCORE_BINARY; ++j) {
    if (i + j < num_data) {
      shared_label[tid] += cuda_labels[i + j];
    }
  }
  __syncthreads();
  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0 && (tid + s) < CALC_INIT_SCORE_BLOCK_SIZE_BINARY) {
      shared_label[tid] += shared_label[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    atomicAdd_system(out_cuda_init_score, shared_label[0]);
  }
}

__global__ void BoostFromScoreKernel_2_BinaryLogloss(double* out_cuda_init_score, const data_size_t num_data, const double sigmoid) {
  const double suml = *out_cuda_init_score;
  const double sumw = static_cast<double>(num_data);
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("******************************************* suml = %f sumw = %f *******************************************\n", suml, sumw);
  }
  const double pavg = suml / sumw;
  const double init_score = log(pavg / (1.0f - pavg)) / sigmoid;
  *out_cuda_init_score = init_score;
}

void CUDABinaryLogloss::LaunchBoostFromScoreKernel() const {
  const data_size_t num_data_per_block = CALC_INIT_SCORE_BLOCK_SIZE_BINARY * NUM_DATA_THREAD_ADD_CALC_INIT_SCORE_BINARY;
  const int num_blocks = (num_data_ + num_data_per_block - 1) / num_data_per_block;
  BoostFromScoreKernel_1_BinaryLogloss<<<num_blocks, CALC_INIT_SCORE_BLOCK_SIZE_BINARY>>>(cuda_label_, num_data_, cuda_boost_from_score_);
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
  BoostFromScoreKernel_2_BinaryLogloss<<<1, 1>>>(cuda_boost_from_score_, num_data_, sigmoid_);
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
}

__global__ void GetGradientsKernel_BinaryLogloss(const double* cuda_scores, const label_t* cuda_labels,
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

void CUDABinaryLogloss::LaunchGetGradientsKernel(const double* scores, score_t* gradients, score_t* hessians) const {
  const int num_blocks = (num_data_ + GET_GRADIENTS_BLOCK_SIZE_BINARY - 1) / GET_GRADIENTS_BLOCK_SIZE_BINARY;
  GetGradientsKernel_BinaryLogloss<<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_BINARY>>>(
    scores,
    cuda_label_,
    sigmoid_,
    num_data_,
    gradients,
    hessians);
}

}  // namespace LightGBM

#endif  // USE_CUDA
