/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_histogram_constructor.hpp"
#include "cuda_leaf_splits_init.hu"

#include <LightGBM/cuda/cuda_utils.h>

namespace LightGBM {

CUDALeafSplitsInit::CUDALeafSplitsInit(const score_t* cuda_gradients,
  const score_t* cuda_hessians, const data_size_t num_data):
cuda_gradients_(cuda_gradients), cuda_hessians_(cuda_hessians), num_data_(num_data) {
  
}

void CUDALeafSplitsInit::Init() {
  num_cuda_blocks_ = 256;

  CUDASUCCESS_OR_FATAL(cudaSetDevice(0));
  CUDASUCCESS_OR_FATAL(cudaMalloc(&smaller_leaf_sum_gradients_, num_cuda_blocks_));
  CUDASUCCESS_OR_FATAL(cudaMalloc(&smaller_leaf_sum_hessians_, num_cuda_blocks_));

  const int num_data_per_blocks = (num_data_ + num_cuda_blocks_ - 1) / num_cuda_blocks_;

  CUDALeafSplitsInitKernel1<<<num_cuda_blocks_, num_data_per_blocks>>>(
    cuda_gradients_, cuda_hessians_, num_data_, smaller_leaf_sum_gradients_,
    smaller_leaf_sum_hessians_);

  CUDALeafSplitsInitKernel2<<<num_cuda_blocks_, num_data_per_blocks>>>(
    cuda_gradients_, cuda_hessians_, num_data_, smaller_leaf_sum_gradients_,
    smaller_leaf_sum_hessians_);
}

}  // namespace LightGBM

#endif  // USE_CUDA
