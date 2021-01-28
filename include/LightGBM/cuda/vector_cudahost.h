/*!
 * Copyright (c) 2020 IBM Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_CUDA_VECTOR_CUDAHOST_H_
#define LIGHTGBM_CUDA_VECTOR_CUDAHOST_H_

#include <LightGBM/utils/common.h>

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#include <stdio.h>

enum LGBM_Device {
  lgbm_device_cpu,
  lgbm_device_gpu,
  lgbm_device_cuda
};

enum Use_Learner {
  use_cpu_learner,
  use_gpu_learner,
  use_cuda_learner
};

namespace LightGBM {

class LGBM_config_ {
 public:
  static int current_device;  // Default: lgbm_device_cpu
  static int current_learner;  // Default: use_cpu_learner
};


template <class T>
struct CHAllocator {
  typedef T value_type;
  CHAllocator() {}
  template <class U> CHAllocator(const CHAllocator<U>& other);
  T* allocate(std::size_t n) {
    T* ptr;
    if (n == 0) return NULL;
    n = (n + kAlignedSize - 1) & -kAlignedSize;
    #ifdef USE_CUDA
      if (LGBM_config_::current_device == lgbm_device_cuda) {
        cudaError_t ret = cudaHostAlloc(&ptr, n*sizeof(T), cudaHostAllocPortable);
        if (ret != cudaSuccess) {
          Log::Warning("Defaulting to malloc in CHAllocator!!!");
          ptr = reinterpret_cast<T*>(_mm_malloc(n*sizeof(T), 16));
        }
      } else {
        ptr = reinterpret_cast<T*>(_mm_malloc(n*sizeof(T), 16));
      }
    #else
      ptr = reinterpret_cast<T*>(_mm_malloc(n*sizeof(T), 16));
    #endif
    return ptr;
  }

  void deallocate(T* p, std::size_t n) {
    (void)n;  // UNUSED
    if (p == NULL) return;
    #ifdef USE_CUDA
      if (LGBM_config_::current_device == lgbm_device_cuda) {
        cudaPointerAttributes attributes;
        cudaPointerGetAttributes(&attributes, p);
        #if CUDA_VERSION >= 10000
          if ((attributes.type == cudaMemoryTypeHost) && (attributes.devicePointer != NULL)) {
            cudaFreeHost(p);
          }
        #else
          if ((attributes.memoryType == cudaMemoryTypeHost) && (attributes.devicePointer != NULL)) {
            cudaFreeHost(p);
          }
        #endif
      } else {
        _mm_free(p);
      }
    #else
      _mm_free(p);
    #endif
  }
};
template <class T, class U>
bool operator==(const CHAllocator<T>&, const CHAllocator<U>&);
template <class T, class U>
bool operator!=(const CHAllocator<T>&, const CHAllocator<U>&);

}  // namespace LightGBM

#endif  // LIGHTGBM_CUDA_VECTOR_CUDAHOST_H_
