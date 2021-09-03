/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include "cuda_xentropy_objective.hpp"

namespace LightGBM {

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

}  // namespace LightGBM
