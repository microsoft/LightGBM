/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_score_updater.hpp"

namespace LightGBM {

__global__ void SetInitScoreKernel(double* cuda_scores, const double* cuda_init_score, const data_size_t num_data) {
  const data_size_t data_index = static_cast<data_size_t>(blockDim.x * blockIdx.x + threadIdx.x);
  const double init_score = *cuda_init_score;
  if (data_index < num_data) {
    cuda_scores[data_index] = init_score;
  }
}

void CUDAScoreUpdater::LaunchSetInitScoreKernel(const double* cuda_init_score) {
  const int num_blocks = (num_data_ + SET_INIT_SCORE_BLOCK_SIZE - 1) / SET_INIT_SCORE_BLOCK_SIZE;
  SetInitScoreKernel<<<num_blocks, SET_INIT_SCORE_BLOCK_SIZE>>>(cuda_scores_, cuda_init_score, num_data_);
}

__global__ void AddScoreKernel(double* cuda_scores, const double* cuda_scores_to_add, const data_size_t num_data) {
  const data_size_t data_index = static_cast<data_size_t>(blockDim.x * blockIdx.x + threadIdx.x);
  if (data_index < num_data) {
    cuda_scores[data_index] += cuda_scores_to_add[data_index];
  }
}

void CUDAScoreUpdater::LaunchAddScoreKernel(const double* cuda_scores_to_add) {
  const int num_blocks = (num_data_ + SET_INIT_SCORE_BLOCK_SIZE - 1) / SET_INIT_SCORE_BLOCK_SIZE;
  AddScoreKernel<<<num_blocks, SET_INIT_SCORE_BLOCK_SIZE>>>(cuda_scores_, cuda_scores_to_add, num_data_);
}

}  // namespace LightGBM

#endif  // USE_CUDA
