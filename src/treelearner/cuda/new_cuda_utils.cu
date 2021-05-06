/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#include "new_cuda_utils.hpp"

namespace LightGBM {

/*template <>
__device__ void PrefixSum<uint32_t>(uint32_t* elements, unsigned int n) {
  unsigned int offset = 1;
  unsigned int threadIdx_x = threadIdx.x;
  for (int d = (n >> 1); d > 0; d >>= 1) {
    if (threadIdx_x < d) {
      const unsigned int src_pos = offset * (2 * threadIdx_x + 1) - 1;
      const unsigned int dst_pos = offset * (2 * threadIdx_x + 2) - 1;
      elements[dst_pos] += elements[src_pos];
    }
    offset <<= 1;
    __syncthreads();
  }
  const uint32_t last_element = elements[n - 1];
  if (threadIdx_x == 0) {
    elements[n - 1] = 0; 
  }
  __syncthreads();
  for (int d = 1; d < n; d <<= 1) {
    if (threadIdx_x < d) {
      const unsigned int dst_pos = offset * (2 * threadIdx_x + 2) - 1;
      const unsigned int src_pos = offset * (2 * threadIdx_x + 1) - 1;
      const uint32_t src_val = elements[src_pos];
      elements[src_pos] = elements[dst_pos];
      elements[dst_pos] += src_val;
    }
    offset >>= 1;
    __syncthreads();
  }
  if (threadIdx_x == 0) {
    elements[n] = elements[n - 1] + last_element;
  }
}*/

}  // namespace LightGBM
