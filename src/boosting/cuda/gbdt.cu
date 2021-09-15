/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include "../gbdt.h"

#define COPY_SUBSAMPLE_GRADIENTS_BLOCK_SIZE (1024)

namespace LightGBM {

__global__ void CopySubsampleGradientsKernel(
  score_t* dst_grad, score_t* dst_hess,
  const score_t* src_grad, const score_t* src_hess,
  const data_size_t* bag_data_indices,
  const data_size_t bag_data_cnt) {
  const data_size_t local_data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (local_data_index < bag_data_cnt) {
    const data_size_t global_data_index = bag_data_indices[local_data_index];
    dst_grad[local_data_index] = src_grad[global_data_index];
    dst_hess[local_data_index] = src_hess[global_data_index];
  }
}

void GBDT::LaunchCopySubsampleGradientsKernel(
score_t* dst_grad, score_t* dst_hess,
const score_t* src_grad, const score_t* src_hess) {
  const int num_blocks = (bag_data_cnt_ + COPY_SUBSAMPLE_GRADIENTS_BLOCK_SIZE - 1) / COPY_SUBSAMPLE_GRADIENTS_BLOCK_SIZE;
  CopySubsampleGradientsKernel<<<num_blocks, COPY_SUBSAMPLE_GRADIENTS_BLOCK_SIZE>>>(
    dst_grad, dst_hess, src_grad, src_hess, bag_data_indices_.data(), bag_data_cnt_);
}

}  // namespace LightGBM
