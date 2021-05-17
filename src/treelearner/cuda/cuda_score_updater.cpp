/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_score_updater.hpp"

namespace LightGBM {

CUDAScoreUpdater::CUDAScoreUpdater(const data_size_t num_data):
num_data_(num_data) {}

void CUDAScoreUpdater::Init() {
  AllocateCUDAMemory<double>(static_cast<size_t>(num_data_), &cuda_scores_);
}

void CUDAScoreUpdater::SetInitScore(const double* cuda_init_score) {
  LaunchSetInitScoreKernel(cuda_init_score);
}

void CUDAScoreUpdater::AddScore(const double* cuda_score_to_add) {
  LaunchAddScoreKernel(cuda_score_to_add);
}

}  // namespace LightGBM

#endif  // USE_CUDA
