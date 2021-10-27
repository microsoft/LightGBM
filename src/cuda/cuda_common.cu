/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifdef USE_CUDA

#include <LightGBM/cuda/cuda_common.hpp>
#include <LightGBM/cuda/cuda_algorithms.hpp>
#include <LightGBM/cuda/cuda_utils.h>

namespace LightGBM {

template <typename T>
__global__ void CalcBitsetLenKernel(const T* vals, int n, size_t* out_len_buffer) {
  __shared__ size_t shared_mem_buffer[32];
  const int i = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  size_t len = 0;
  if (i < n) {
    const T val = vals[i];
    len = (val / 32) + 1;
  }
  const size_t block_max_len = ShuffleReduceMax<T>(len, shared_mem_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    out_len_buffer[blockIdx.x] = block_max_len;
  }
}

__global__ void ReduceBlockMaxLen(const size_t* out_len_buffer, const int num_blocks) {
  __shared__ size_t shared_mem_buffer[32];
  size_t max_len = 0;
  for (int i = static_cast<int>(threadIdx.x); i < num_blocks; i += static_cast<int>(blockDim.x)) {
    max_len = max(out_len_buffer[i]);
  }
  const all_max_len = ShuffleReduceMax<size_t>(max_len, shared_mem_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    out_len_buffer[0] = max_len;
  }
}

template <typename T>
__global__ void CUDAConstructBitsetKernel(const T* vals, int n, uint32_t* out) {
  const int i = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  if (i < n) {
    const T val = vals[i];
    out[val / 32] |= (0x1 << (val % 32));
  }
}

template <typename T>
void CUDAConstructBitsetInner(const T* vals, int n, uint32_t* out) {
  const int num_blocks = (n + NUM_THREADS_PER_BLOCK_CUDA_COMMON - 1) / NUM_THREADS_PER_BLOCK_CUDA_COMMON;
  CUDAConstructBitsetKernel<T><<<num_blocks, NUM_THREADS_PER_BLOCK_CUDA_COMMON>>>(vals, n, out);
}

template <typename T>
size_t CUDABitsetLenInner(const T* vals, int n, size_t* out_len_buffer) {
  const int num_blocks = (n + NUM_THREADS_PER_BLOCK_CUDA_COMMON - 1) / NUM_THREADS_PER_BLOCK_CUDA_COMMON;
  CalcBitsetLenKernel<T><<<num_blocks, NUM_THREADS_PER_BLOCK_CUDA_COMMON>>>(vals, n, out_len_buffer);
  ReduceBlockMaxLen<<<1, NUM_THREADS_PER_BLOCK_CUDA_COMMON>>>(out_len_buffer, num_blocks);
  size_t host_max_len = 0;
  CopyFromCUDADeviceToHost<size_t>(&host_max_len, out_len_buffer, 1, __FILE__, __LINE__);
  return host_max_len;
}

}  // namespace LightGBM

#endif  // USE_CUDA
