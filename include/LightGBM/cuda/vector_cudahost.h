/*
 * ibmGBT: IBM CUDA Accelerated LightGBM 
 *
 * IBM Confidential
 * (C) Copyright IBM Corp. 2019. All Rights Reserved.
 *
 * The source code for this program is not published or otherwise
 * divested of its trade secrets, irrespective of what has been
 * deposited with the U.S. Copyright Office.
 *
 * US Government Users Restricted Rights - Use, duplication or
 * disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
 */

#ifndef LGBM_CUDA_VECTOR_CH_H
#define LGBM_CUDA_VECTOR_CH_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

//LGBM_CUDA

namespace LightGBM {

#define lgbm_device_cpu 0
#define lgbm_device_gpu 1
#define lgbm_device_cuda 2

#define use_cpu_learner 0
#define use_gpu_learner 1
#define use_cuda_learner 2

class LGBM_config_ {
 public:
  static int current_device; // Default: lgbm_device_cpu 
  static int current_learner; // Default: use_cpu_learner
};

} // namespace LightGBM


template <class T>
struct CHAllocator {
 typedef T value_type;
 CHAllocator() {}
 template <class U> CHAllocator(const CHAllocator<U>& other);
 T* allocate(std::size_t n)
 {
   T* ptr;
   if (n == 0) return NULL;
   #ifdef USE_CUDA
      if (LightGBM::LGBM_config_::current_device == lgbm_device_cuda){
          cudaError_t ret= cudaHostAlloc(&ptr, n*sizeof(T), cudaHostAllocPortable);
          if (ret != cudaSuccess){
             ptr = (T*) malloc(n*sizeof(T));
          }
      }
      else{
            ptr = (T*) malloc(n*sizeof(T));
      }
   #else
      ptr = (T*) malloc(n*sizeof(T));
   #endif
   return ptr;
 }

 void deallocate(T* p, std::size_t n)
 {
    if (p==NULL) return;
    #ifdef USE_CUDA
      if (LightGBM::LGBM_config_::current_device == lgbm_device_cuda){
          cudaPointerAttributes attributes;
          cudaPointerGetAttributes (&attributes, p);
          if ((attributes.type == cudaMemoryTypeHost) && (attributes.devicePointer != NULL)){
              cudaFreeHost(p);
          }
      } 
      else{
        free(p);
      }
    #else
        free(p);
    #endif
 }

};
template <class T, class U>
bool operator==(const CHAllocator<T>&, const CHAllocator<U>&);
template <class T, class U>
bool operator!=(const CHAllocator<T>&, const CHAllocator<U>&);

#endif
