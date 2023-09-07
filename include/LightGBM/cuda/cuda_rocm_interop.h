/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifdef USE_CUDA

#ifdef __HIP_PLATFORM_AMD__
// ROCm doesn't have __shfl_down_sync, only __shfl_down without mask.
// Since mask is full 0xffffffff, we can use __shfl_down instead.
#define __shfl_down_sync(mask, val, offset) __shfl_down(val, offset)
#define __shfl_up_sync(mask, val, offset) __shfl_up(val, offset)
// ROCm warpSize is constexpr and is either 32 or 64 depending on gfx arch.
#define WARPSIZE warpSize
#else
// CUDA warpSize is not a constexpr, but always 32
#define WARPSIZE 32
#endif

#endif
