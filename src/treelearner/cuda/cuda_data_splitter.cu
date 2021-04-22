/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#include "cuda_data_splitter.hpp"

#define FILL_INDICES_BLOCK_SIZE (1024)

namespace LightGBM {

__global__ void FillDataIndicesBeforeTrainKernel(const data_size_t* cuda_num_data,
  data_size_t* data_indices) {
  const data_size_t num_data_ref = *cuda_num_data;
  const unsigned int data_index = threadIdx.x + blockIdx.x * blockDim.x;
  if (data_index < num_data_ref) {
    data_indices[data_index] = data_index;
  }
}

void CUDADataSplitter::LaunchFillDataIndicesBeforeTrain() {
  const int num_blocks = (num_data_ + FILL_INDICES_BLOCK_SIZE - 1) / FILL_INDICES_BLOCK_SIZE;
  FillDataIndicesBeforeTrainKernel<<<num_blocks, FILL_INDICES_BLOCK_SIZE>>>(cuda_num_data_, cuda_data_indices_); 
}

}  // namespace LightGBM