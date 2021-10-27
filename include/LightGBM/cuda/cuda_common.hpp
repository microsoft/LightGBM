/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifndef LIGHTGBM_CUDA_CUDA_COMMON_HPP_
#define LIGHTGBM_CUDA_CUDA_COMMON_HPP_

#ifdef USE_CUDA

#define NUM_THREADS_PER_BLOCK_CUDA_COMMON (1024)

#include <LightGBM/cuda/cuda_split_info.hpp>

namespace LightGBM {

size_t CUDABitsetLen(const CUDASplitInfo* split_info, size_t* out_len_buffer);

void CUDAConstructBitset(const CUDASplitInfo* split_info, uint32_t* out);

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_CUDA_CUDA_COMMON_HPP_
