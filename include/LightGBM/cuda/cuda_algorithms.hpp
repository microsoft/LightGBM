/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifndef LIGHTGBM_CUDA_CUDA_ALGORITHMS_HPP_
#define LIGHTGBM_CUDA_CUDA_ALGORITHMS_HPP_

#ifdef USE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <LightGBM/bin.h>
#include <LightGBM/utils/log.h>

#define NUM_BANKS_DATA_PARTITION (16)
#define LOG_NUM_BANKS_DATA_PARTITION (4)

#define CONFLICT_FREE_INDEX(n) \
  ((n) + ((n) >> LOG_NUM_BANKS_DATA_PARTITION)) \

namespace LightGBM {

template <typename T>
__device__ void ReduceSum(T* values, size_t n);

template <typename T>
__device__ void ReduceSumConflictFree(T* values, size_t n);

template <typename T>
__device__ void ReduceMax(T* values, size_t n);

template <typename T>
__device__ void PrefixSum(T* values, size_t n);

template <typename T>
__device__ void PrefixSumConflictFree(T* values, size_t n);

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_CUDA_CUDA_ALGORITHMS_HPP_
