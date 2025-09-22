/*!
 * Copyright(C) 2023 Advanced Micro Devices, Inc. All rights reserved.
 */
#pragma once

#if defined(USE_CUDA) || defined(USE_ROCM)

#if defined(__HIP_PLATFORM_AMD__)

// ROCm doesn't have __shfl_down_sync, only __shfl_down without mask.
// Since mask is full 0xffffffff, we can use __shfl_down instead.
#define __shfl_down_sync(mask, val, offset) __shfl_down(val, offset)
#define __shfl_up_sync(mask, val, offset) __shfl_up(val, offset)

// ROCm doesn't have atomicAdd_block, but it should be semantically the same as atomicAdd
#define atomicAdd_block atomicAdd

// hipify
#include <hip/hip_runtime.h>
#define cudaDeviceProp hipDeviceProp_t
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaError_t hipError_t
#define cudaFree hipFree
#define cudaFreeHost hipFreeHost
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetErrorName hipGetErrorName
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaHostAlloc hipHostAlloc
#define cudaHostAllocPortable hipHostAllocPortable
#define cudaMalloc hipMalloc
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemoryTypeHost hipMemoryTypeHost
#define cudaMemset hipMemset
#define cudaPointerAttributes hipPointerAttribute_t
#define cudaPointerGetAttributes hipPointerGetAttributes
#define cudaSetDevice hipSetDevice
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess

// warpSize is only allowed for device code.
// HIP header used to define warpSize as a constexpr that was either 32 or 64
// depending on the target device, and then always set it to 64 for host code.
static inline constexpr int WARP_SIZE_INTERNAL() {
#if defined(__GFX9__)
  return 64;
#else // __GFX9__
  return 32;
#endif // __GFX9__
}
#define WARPSIZE (WARP_SIZE_INTERNAL())

#else // __HIP_PLATFORM_AMD__
// CUDA warpSize is not a constexpr, but always 32
#define WARPSIZE 32
#endif

#endif // USE_CUDA || USE_ROCM
