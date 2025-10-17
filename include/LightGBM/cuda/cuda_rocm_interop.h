/*!
 * Copyright(C) 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef LIGHTGBM_INCLUDE_LIGHTGBM_CUDA_CUDA_ROCM_INTEROP_H_
#define LIGHTGBM_INCLUDE_LIGHTGBM_CUDA_CUDA_ROCM_INTEROP_H_

#ifdef USE_CUDA

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
// ROCm doesn't have __shfl_down_sync, only __shfl_down without mask.
// Since mask is full 0xffffffff, we can use __shfl_down instead.
#define __shfl_down_sync(mask, val, offset) __shfl_down(val, offset)
#define __shfl_up_sync(mask, val, offset) __shfl_up(val, offset)
// ROCm warpSize is constexpr and is either 32 or 64 depending on gfx arch.
#define WARPSIZE warpSize
// ROCm doesn't have atomicAdd_block, but it should be semantically the same as atomicAdd
#define atomicAdd_block atomicAdd
#else
// CUDA warpSize is not a constexpr, but always 32
#define WARPSIZE 32
#endif  // defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)

#endif  // USE_CUDA

#endif  // LIGHTGBM_INCLUDE_LIGHTGBM_CUDA_CUDA_ROCM_INTEROP_H_
