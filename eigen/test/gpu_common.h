#ifndef EIGEN_TEST_GPU_COMMON_H
#define EIGEN_TEST_GPU_COMMON_H

#ifdef EIGEN_USE_HIP
  #include <hip/hip_runtime.h>
  #include <hip/hip_runtime_api.h>
#else
  #include <cuda.h>
  #include <cuda_runtime.h>
  #include <cuda_runtime_api.h>
#endif

#include <iostream>

#define EIGEN_USE_GPU
#include <unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h>

#if !defined(__CUDACC__) && !defined(__HIPCC__)
dim3 threadIdx, blockDim, blockIdx;
#endif

template<typename Kernel, typename Input, typename Output>
void run_on_cpu(const Kernel& ker, int n, const Input& in, Output& out)
{
  for(int i=0; i<n; i++)
    ker(i, in.data(), out.data());
}


template<typename Kernel, typename Input, typename Output>
__global__
__launch_bounds__(1024)
void run_on_gpu_meta_kernel(const Kernel ker, int n, const Input* in, Output* out)
{
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if(i<n) {
    ker(i, in, out);
  }
}


template<typename Kernel, typename Input, typename Output>
void run_on_gpu(const Kernel& ker, int n, const Input& in, Output& out)
{
  typename Input::Scalar*  d_in;
  typename Output::Scalar* d_out;
  std::ptrdiff_t in_bytes  = in.size()  * sizeof(typename Input::Scalar);
  std::ptrdiff_t out_bytes = out.size() * sizeof(typename Output::Scalar);
  
  gpuMalloc((void**)(&d_in),  in_bytes);
  gpuMalloc((void**)(&d_out), out_bytes);
  
  gpuMemcpy(d_in,  in.data(),  in_bytes,  gpuMemcpyHostToDevice);
  gpuMemcpy(d_out, out.data(), out_bytes, gpuMemcpyHostToDevice);
  
  // Simple and non-optimal 1D mapping assuming n is not too large
  // That's only for unit testing!
  dim3 Blocks(128);
  dim3 Grids( (n+int(Blocks.x)-1)/int(Blocks.x) );

  gpuDeviceSynchronize();
  
#ifdef EIGEN_USE_HIP
  hipLaunchKernelGGL(HIP_KERNEL_NAME(run_on_gpu_meta_kernel<Kernel,
				     typename std::decay<decltype(*d_in)>::type,
				     typename std::decay<decltype(*d_out)>::type>), 
		     dim3(Grids), dim3(Blocks), 0, 0, ker, n, d_in, d_out);
#else
  run_on_gpu_meta_kernel<<<Grids,Blocks>>>(ker, n, d_in, d_out);
#endif
  
  gpuDeviceSynchronize();
  
  // check inputs have not been modified
  gpuMemcpy(const_cast<typename Input::Scalar*>(in.data()),  d_in,  in_bytes,  gpuMemcpyDeviceToHost);
  gpuMemcpy(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost);
  
  gpuFree(d_in);
  gpuFree(d_out);
}


template<typename Kernel, typename Input, typename Output>
void run_and_compare_to_gpu(const Kernel& ker, int n, const Input& in, Output& out)
{
  Input  in_ref,  in_gpu;
  Output out_ref, out_gpu;
  #if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
  in_ref = in_gpu = in;
  out_ref = out_gpu = out;
  #else
  EIGEN_UNUSED_VARIABLE(in);
  EIGEN_UNUSED_VARIABLE(out);
  #endif
  run_on_cpu (ker, n, in_ref,  out_ref);
  run_on_gpu(ker, n, in_gpu, out_gpu);
  #if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
  VERIFY_IS_APPROX(in_ref, in_gpu);
  VERIFY_IS_APPROX(out_ref, out_gpu);
  #endif
}

struct compile_time_device_info {
  EIGEN_DEVICE_FUNC
  void operator()(int /*i*/, const int* /*in*/, int* info) const
  {
    #if defined(__CUDA_ARCH__)
    info[0] = int(__CUDA_ARCH__ +0);
    #endif
    #if defined(EIGEN_HIP_DEVICE_COMPILE)
    info[1] = int(EIGEN_HIP_DEVICE_COMPILE +0);
    #endif
  }
};

void ei_test_init_gpu()
{
  int device = 0;
  gpuDeviceProp_t deviceProp;
  gpuGetDeviceProperties(&deviceProp, device);

  ArrayXi dummy(1), info(10);
  info = -1;
  run_on_gpu(compile_time_device_info(),10,dummy,info);


  std::cout << "GPU compile-time info:\n";
  
  #ifdef EIGEN_CUDACC
  std::cout << "  EIGEN_CUDACC:                 " << int(EIGEN_CUDACC) << "\n";
  #endif
  
  #ifdef EIGEN_CUDA_SDK_VER
  std::cout << "  EIGEN_CUDA_SDK_VER:             " << int(EIGEN_CUDA_SDK_VER) << "\n";
  #endif

  #ifdef EIGEN_COMP_NVCC
  std::cout << "  EIGEN_COMP_NVCC:             " << int(EIGEN_COMP_NVCC) << "\n";
  #endif
  
  #ifdef EIGEN_HIPCC
  std::cout << "  EIGEN_HIPCC:                 " << int(EIGEN_HIPCC) << "\n";
  #endif

  std::cout << "  EIGEN_CUDA_ARCH:             " << info[0] << "\n";  
  std::cout << "  EIGEN_HIP_DEVICE_COMPILE:    " << info[1] << "\n";

  std::cout << "GPU device info:\n";
  std::cout << "  name:                        " << deviceProp.name << "\n";
  std::cout << "  capability:                  " << deviceProp.major << "." << deviceProp.minor << "\n";
  std::cout << "  multiProcessorCount:         " << deviceProp.multiProcessorCount << "\n";
  std::cout << "  maxThreadsPerMultiProcessor: " << deviceProp.maxThreadsPerMultiProcessor << "\n";
  std::cout << "  warpSize:                    " << deviceProp.warpSize << "\n";
  std::cout << "  regsPerBlock:                " << deviceProp.regsPerBlock << "\n";
  std::cout << "  concurrentKernels:           " << deviceProp.concurrentKernels << "\n";
  std::cout << "  clockRate:                   " << deviceProp.clockRate << "\n";
  std::cout << "  canMapHostMemory:            " << deviceProp.canMapHostMemory << "\n";
  std::cout << "  computeMode:                 " << deviceProp.computeMode << "\n";
}

#endif // EIGEN_TEST_GPU_COMMON_H
