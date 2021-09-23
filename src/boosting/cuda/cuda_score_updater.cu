/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include "cuda_score_updater.hpp"

namespace LightGBM {

__global__ void AddScoreConstantKernel(
  const double val,
  const size_t offset,
  const data_size_t num_data,
  double* score) {
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (data_index < num_data) {
    score[data_index + offset] += val;
  }
}

void CUDAScoreUpdater::LaunchAddScoreConstantKernel(const double val, const size_t offset) {
  const int num_blocks = (num_data_ + num_threads_per_block_) / num_threads_per_block_;
  Log::Warning("adding init score = %f", val);
  AddScoreConstantKernel<<<num_blocks, num_threads_per_block_>>>(val, offset, num_data_, cuda_score_);
}

__global__ void MultiplyScoreConstantKernel(
  const double val,
  const size_t offset,
  const data_size_t num_data,
  double* score) {
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (data_index < num_data) {
    score[data_index] *= val;
  }
}

void CUDAScoreUpdater::LaunchMultiplyScoreConstantKernel(const double val, const size_t offset) {
  const int num_blocks = (num_data_ + num_threads_per_block_) / num_threads_per_block_;
  MultiplyScoreConstantKernel<<<num_blocks, num_threads_per_block_>>>(val, offset, num_data_, cuda_score_);
}

}
