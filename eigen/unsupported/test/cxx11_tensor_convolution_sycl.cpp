// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016
// Mehdi Goli    Codeplay Software Ltd.
// Ralph Potter  Codeplay Software Ltd.
// Luke Iwanski  Codeplay Software Ltd.
// Contact: <eigen@codeplay.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_NO_COMPLEX

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int64_t
#define EIGEN_USE_SYCL

#include <iostream>
#include <chrono>
#include <ctime>

#include "main.h"
#include <unsupported/Eigen/CXX11/Tensor>
#include <iomanip>

using Eigen::array;
using Eigen::SyclDevice;
using Eigen::Tensor;
using Eigen::TensorMap;
static const float error_threshold =1e-4f;


template <typename DataType, int DataLayout, typename IndexType>
static void test_larg_expr1D(const Eigen::SyclDevice& sycl_device)
{
  IndexType indim0 =53;
  IndexType indim1= 55;
  IndexType indim2= 51;
  IndexType outdim0=50;
  IndexType outdim1=55;
  IndexType outdim2=51;
  Eigen::array<IndexType, 3> input_dims = {{indim0, indim1, indim2}};
  Eigen::array<IndexType, 1> kernel_dims = {{4}};
  Eigen::array<IndexType, 3> result_dims = {{outdim0, outdim1, outdim2}};

  Tensor<DataType, 3, DataLayout, IndexType> input(input_dims);
  Tensor<DataType, 1, DataLayout,IndexType> kernel(kernel_dims);
  Tensor<DataType, 3, DataLayout,IndexType> result(result_dims);
  Tensor<DataType, 3, DataLayout,IndexType> result_host(result_dims);

  Eigen::array<IndexType, 1> dims3{{0}};

  input.setRandom();
  kernel.setRandom();
  result.setZero();
  result_host.setZero();

  std::size_t input_bytes = input.size()  * sizeof(DataType);
  std::size_t kernel_bytes = kernel.size() * sizeof(DataType);
  std::size_t result_bytes = result.size() * sizeof(DataType);

  DataType * d_input  = static_cast<DataType*>(sycl_device.allocate(input_bytes));
  DataType * d_kernel  = static_cast<DataType*>(sycl_device.allocate(kernel_bytes));
  DataType * d_result =  static_cast<DataType*>(sycl_device.allocate(result_bytes));

  Eigen::TensorMap<Eigen::Tensor<DataType, 3, DataLayout, IndexType> > gpu_input(d_input, input_dims);
  Eigen::TensorMap<Eigen::Tensor<DataType, 1, DataLayout, IndexType> > gpu_kernel(d_kernel, kernel_dims);
  Eigen::TensorMap<Eigen::Tensor<DataType, 3, DataLayout, IndexType> > gpu_result(d_result, result_dims);
  sycl_device.memcpyHostToDevice(d_input, input.data(), input_bytes);
  sycl_device.memcpyHostToDevice(d_kernel, kernel.data(), kernel_bytes);

  gpu_result.device(sycl_device)=gpu_input.convolve(gpu_kernel, dims3);
  sycl_device.memcpyDeviceToHost(result.data(), d_result, result_bytes);

  result_host=input.convolve(kernel, dims3);

for(IndexType i=0; i< outdim0; i++ ){
  for(IndexType j=0; j< outdim1; j++ ){
    for(IndexType k=0; k< outdim2; k++ ){
      if (!(Eigen::internal::isApprox(result(i,j,k), result_host(i,j,k), error_threshold))) {
        std::cout <<std::setprecision(16)<< "mismatch detected at index  ( "<< i  << " , "  << j  << ", " << k << " ) " << " \t " << result(i,j,k) << " vs "<<  result_host(i,j,k) << std::endl;
        assert(false);
      }
    }
  }
}
  sycl_device.deallocate(d_input);
  sycl_device.deallocate(d_kernel);
  sycl_device.deallocate(d_result);

}


template <typename DataType, int DataLayout, typename IndexType>
static void test_larg_expr2D(const Eigen::SyclDevice& sycl_device)
{
  IndexType indim0 =53;
  IndexType indim1= 55;
  IndexType indim2= 51;
  IndexType outdim0=50;
  IndexType outdim1=51;
  IndexType outdim2=51;
  Eigen::array<IndexType, 3> input_dims = {{indim0, indim1, indim2}};
  Eigen::array<IndexType, 2> kernel_dims = {{4,5}};
  Eigen::array<IndexType, 3> result_dims = {{outdim0, outdim1, outdim2}};

  Tensor<DataType, 3, DataLayout, IndexType> input(input_dims);
  Tensor<DataType, 2, DataLayout,IndexType> kernel(kernel_dims);
  Tensor<DataType, 3, DataLayout,IndexType> result(result_dims);
  Tensor<DataType, 3, DataLayout,IndexType> result_host(result_dims);

  Eigen::array<IndexType, 2> dims3{{0,1}};

  input.setRandom();
  kernel.setRandom();
  result.setZero();
  result_host.setZero();

  std::size_t input_bytes = input.size()  * sizeof(DataType);
  std::size_t kernel_bytes = kernel.size() * sizeof(DataType);
  std::size_t result_bytes = result.size() * sizeof(DataType);

  DataType * d_input  = static_cast<DataType*>(sycl_device.allocate(input_bytes));
  DataType * d_kernel  = static_cast<DataType*>(sycl_device.allocate(kernel_bytes));
  DataType * d_result =  static_cast<DataType*>(sycl_device.allocate(result_bytes));

  Eigen::TensorMap<Eigen::Tensor<DataType, 3, DataLayout, IndexType> > gpu_input(d_input, input_dims);
  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType> > gpu_kernel(d_kernel, kernel_dims);
  Eigen::TensorMap<Eigen::Tensor<DataType, 3, DataLayout, IndexType> > gpu_result(d_result, result_dims);
  sycl_device.memcpyHostToDevice(d_input, input.data(), input_bytes);
  sycl_device.memcpyHostToDevice(d_kernel, kernel.data(), kernel_bytes);

  gpu_result.device(sycl_device)=gpu_input.convolve(gpu_kernel, dims3);
  sycl_device.memcpyDeviceToHost(result.data(), d_result, result_bytes);

  result_host=input.convolve(kernel, dims3);

for(IndexType i=0; i< outdim0; i++ ){
  for(IndexType j=0; j< outdim1; j++ ){
    for(IndexType k=0; k< outdim2; k++ ){
      if (!(Eigen::internal::isApprox(result(i,j,k), result_host(i,j,k), error_threshold))) {
        std::cout <<std::setprecision(16)<< "mismatch detected at index  ( "<< i  << " , "  << j  << ", " << k << " ) " << " \t " << result(i,j,k) << " vs "<<  result_host(i,j,k) << std::endl;
        assert(false);
      }
    }
  }
}
  sycl_device.deallocate(d_input);
  sycl_device.deallocate(d_kernel);
  sycl_device.deallocate(d_result);

}


template <typename DataType, int DataLayout, typename IndexType>
static void test_larg_expr3D(const Eigen::SyclDevice& sycl_device)
{
  IndexType indim0 =53;
  IndexType indim1= 55;
  IndexType indim2= 51;
  IndexType outdim0=50;
  IndexType outdim1=51;
  IndexType outdim2=49;
  Eigen::array<IndexType, 3> input_dims = {{indim0, indim1, indim2}};
  Eigen::array<IndexType, 3> kernel_dims = {{4,5,3}};
  Eigen::array<IndexType, 3> result_dims = {{outdim0, outdim1, outdim2}};

  Tensor<DataType, 3, DataLayout, IndexType> input(input_dims);
  Tensor<DataType, 3, DataLayout,IndexType> kernel(kernel_dims);
  Tensor<DataType, 3, DataLayout,IndexType> result(result_dims);
  Tensor<DataType, 3, DataLayout,IndexType> result_host(result_dims);

  Eigen::array<IndexType, 3> dims3{{0,1,2}};

  input.setRandom();
  kernel.setRandom();
  result.setZero();
  result_host.setZero();

  std::size_t input_bytes = input.size()  * sizeof(DataType);
  std::size_t kernel_bytes = kernel.size() * sizeof(DataType);
  std::size_t result_bytes = result.size() * sizeof(DataType);

  DataType * d_input  = static_cast<DataType*>(sycl_device.allocate(input_bytes));
  DataType * d_kernel  = static_cast<DataType*>(sycl_device.allocate(kernel_bytes));
  DataType * d_result =  static_cast<DataType*>(sycl_device.allocate(result_bytes));

  Eigen::TensorMap<Eigen::Tensor<DataType, 3, DataLayout, IndexType> > gpu_input(d_input, input_dims);
  Eigen::TensorMap<Eigen::Tensor<DataType, 3, DataLayout, IndexType> > gpu_kernel(d_kernel, kernel_dims);
  Eigen::TensorMap<Eigen::Tensor<DataType, 3, DataLayout, IndexType> > gpu_result(d_result, result_dims);
  sycl_device.memcpyHostToDevice(d_input, input.data(), input_bytes);
  sycl_device.memcpyHostToDevice(d_kernel, kernel.data(), kernel_bytes);

  gpu_result.device(sycl_device)=gpu_input.convolve(gpu_kernel, dims3);
  sycl_device.memcpyDeviceToHost(result.data(), d_result, result_bytes);

  result_host=input.convolve(kernel, dims3);

for(IndexType i=0; i< outdim0; i++ ){
  for(IndexType j=0; j< outdim1; j++ ){
    for(IndexType k=0; k< outdim2; k++ ){
      if (!(Eigen::internal::isApprox(result(i,j,k), result_host(i,j,k), error_threshold))) {
        std::cout <<std::setprecision(16)<< "mismatch detected at index  ( "<< i  << " , "  << j  << ", " << k << " ) " << " \t " << result(i,j,k) << " vs "<<  result_host(i,j,k) << std::endl;
        assert(false);
      }
    }
  }
}
  sycl_device.deallocate(d_input);
  sycl_device.deallocate(d_kernel);
  sycl_device.deallocate(d_result);

}


template <typename DataType, int DataLayout, typename IndexType>
static void test_evals(const Eigen::SyclDevice& sycl_device)
{
  Eigen::array<IndexType, 2> input_dims = {{3, 3}};
  Eigen::array<IndexType, 1> kernel_dims = {{2}};
  Eigen::array<IndexType, 2> result_dims = {{2, 3}};

  Tensor<DataType, 2, DataLayout, IndexType> input(input_dims);
  Tensor<DataType, 1, DataLayout,IndexType> kernel(kernel_dims);
  Tensor<DataType, 2, DataLayout,IndexType> result(result_dims);

  Eigen::array<IndexType, 1> dims3{{0}};

  input.setRandom();
  kernel.setRandom();
  result.setZero();

  std::size_t input_bytes = input.size()  * sizeof(DataType);
  std::size_t kernel_bytes = kernel.size() * sizeof(DataType);
  std::size_t result_bytes = result.size() * sizeof(DataType);

  DataType * d_input  = static_cast<DataType*>(sycl_device.allocate(input_bytes));
  DataType * d_kernel  = static_cast<DataType*>(sycl_device.allocate(kernel_bytes));
  DataType * d_result =  static_cast<DataType*>(sycl_device.allocate(result_bytes));

  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType> > gpu_input(d_input, input_dims);
  Eigen::TensorMap<Eigen::Tensor<DataType, 1, DataLayout, IndexType> > gpu_kernel(d_kernel, kernel_dims);
  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType> > gpu_result(d_result, result_dims);
  sycl_device.memcpyHostToDevice(d_input, input.data(), input_bytes);
  sycl_device.memcpyHostToDevice(d_kernel, kernel.data(), kernel_bytes);

  gpu_result.device(sycl_device)=gpu_input.convolve(gpu_kernel, dims3);
  sycl_device.memcpyDeviceToHost(result.data(), d_result, result_bytes);

  VERIFY_IS_APPROX(result(0,0), input(0,0)*kernel(0) + input(1,0)*kernel(1));  // index 0
  VERIFY_IS_APPROX(result(0,1), input(0,1)*kernel(0) + input(1,1)*kernel(1));  // index 2
  VERIFY_IS_APPROX(result(0,2), input(0,2)*kernel(0) + input(1,2)*kernel(1));  // index 4
  VERIFY_IS_APPROX(result(1,0), input(1,0)*kernel(0) + input(2,0)*kernel(1));  // index 1
  VERIFY_IS_APPROX(result(1,1), input(1,1)*kernel(0) + input(2,1)*kernel(1));  // index 3
  VERIFY_IS_APPROX(result(1,2), input(1,2)*kernel(0) + input(2,2)*kernel(1));  // index 5

  sycl_device.deallocate(d_input);
  sycl_device.deallocate(d_kernel);
  sycl_device.deallocate(d_result);
}

template <typename DataType, int DataLayout, typename IndexType>
static void test_expr(const Eigen::SyclDevice& sycl_device)
{
  Eigen::array<IndexType, 2> input_dims = {{3, 3}};
  Eigen::array<IndexType, 2> kernel_dims = {{2, 2}};
  Eigen::array<IndexType, 2> result_dims = {{2, 2}};

  Tensor<DataType, 2, DataLayout, IndexType> input(input_dims);
  Tensor<DataType, 2, DataLayout, IndexType> kernel(kernel_dims);
  Tensor<DataType, 2, DataLayout, IndexType> result(result_dims);

  input.setRandom();
  kernel.setRandom();
  Eigen::array<IndexType, 2> dims;
  dims[0] = 0;
  dims[1] = 1;

  std::size_t input_bytes = input.size()  * sizeof(DataType);
  std::size_t kernel_bytes = kernel.size() * sizeof(DataType);
  std::size_t result_bytes = result.size() * sizeof(DataType);

  DataType * d_input  = static_cast<DataType*>(sycl_device.allocate(input_bytes));
  DataType * d_kernel  = static_cast<DataType*>(sycl_device.allocate(kernel_bytes));
  DataType * d_result =  static_cast<DataType*>(sycl_device.allocate(result_bytes));

  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout,IndexType> > gpu_input(d_input, input_dims);
  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout,IndexType> > gpu_kernel(d_kernel, kernel_dims);
  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout,IndexType> > gpu_result(d_result, result_dims);
  sycl_device.memcpyHostToDevice(d_input, input.data(), input_bytes);
  sycl_device.memcpyHostToDevice(d_kernel, kernel.data(), kernel_bytes);

  gpu_result.device(sycl_device)=gpu_input.convolve(gpu_kernel, dims);
  sycl_device.memcpyDeviceToHost(result.data(), d_result, result_bytes);

  VERIFY_IS_APPROX(result(0,0), input(0,0)*kernel(0,0) + input(0,1)*kernel(0,1) +
                                input(1,0)*kernel(1,0) + input(1,1)*kernel(1,1));
  VERIFY_IS_APPROX(result(0,1), input(0,1)*kernel(0,0) + input(0,2)*kernel(0,1) +
                                input(1,1)*kernel(1,0) + input(1,2)*kernel(1,1));
  VERIFY_IS_APPROX(result(1,0), input(1,0)*kernel(0,0) + input(1,1)*kernel(0,1) +
                                input(2,0)*kernel(1,0) + input(2,1)*kernel(1,1));
  VERIFY_IS_APPROX(result(1,1), input(1,1)*kernel(0,0) + input(1,2)*kernel(0,1) +
                                input(2,1)*kernel(1,0) + input(2,2)*kernel(1,1));

  sycl_device.deallocate(d_input);
  sycl_device.deallocate(d_kernel);
  sycl_device.deallocate(d_result);
}


template <typename DataType, int DataLayout, typename IndexType>
static void test_modes(const Eigen::SyclDevice& sycl_device){

Eigen::array<IndexType, 1> input_dims = {{3}};
Eigen::array<IndexType, 1> kernel_dims = {{3}};

Tensor<DataType, 1, DataLayout, IndexType> input(input_dims);
Tensor<DataType, 1, DataLayout, IndexType> kernel(kernel_dims);

input.setRandom();
kernel.setRandom();
Eigen::array<IndexType, 1> dims;
dims[0] = 0;

  input(0) = 1.0f;
  input(1) = 2.0f;
  input(2) = 3.0f;
  kernel(0) = 0.5f;
  kernel(1) = 1.0f;
  kernel(2) = 0.0f;

  Eigen::array<std::pair<IndexType, IndexType>, 1> padding;

  // Emulate VALID mode (as defined in
  // http://docs.scipy.org/doc/numpy/reference/generated/numpy.convolve.html).
  padding[0] = std::make_pair(0, 0);
  Tensor<DataType, 1, DataLayout, IndexType> valid(1);

  std::size_t input_bytes = input.size()  * sizeof(DataType);
  std::size_t kernel_bytes = kernel.size() * sizeof(DataType);
  std::size_t valid_bytes = valid.size() * sizeof(DataType);

  DataType * d_input  = static_cast<DataType*>(sycl_device.allocate(input_bytes));
  DataType * d_kernel  = static_cast<DataType*>(sycl_device.allocate(kernel_bytes));
  DataType * d_valid =  static_cast<DataType*>(sycl_device.allocate(valid_bytes));

  Eigen::TensorMap<Eigen::Tensor<DataType, 1, DataLayout,IndexType> > gpu_input(d_input, input_dims);
  Eigen::TensorMap<Eigen::Tensor<DataType, 1, DataLayout,IndexType> > gpu_kernel(d_kernel, kernel_dims);
  Eigen::TensorMap<Eigen::Tensor<DataType, 1, DataLayout,IndexType> > gpu_valid(d_valid, valid.dimensions());
  sycl_device.memcpyHostToDevice(d_input, input.data(), input_bytes);
  sycl_device.memcpyHostToDevice(d_kernel, kernel.data(), kernel_bytes);

  gpu_valid.device(sycl_device)=gpu_input.pad(padding).convolve(gpu_kernel, dims);
  sycl_device.memcpyDeviceToHost(valid.data(), d_valid, valid_bytes);

  VERIFY_IS_EQUAL(valid.dimension(0), 1);
  VERIFY_IS_APPROX(valid(0), 2.5f);

  // Emulate SAME mode (as defined in
  // http://docs.scipy.org/doc/numpy/reference/generated/numpy.convolve.html).
  padding[0] = std::make_pair(1, 1);
  Tensor<DataType, 1, DataLayout, IndexType> same(3);
  std::size_t same_bytes = same.size() * sizeof(DataType);
  DataType * d_same =  static_cast<DataType*>(sycl_device.allocate(same_bytes));
  Eigen::TensorMap<Eigen::Tensor<DataType, 1, DataLayout,IndexType> > gpu_same(d_same, same.dimensions());
  gpu_same.device(sycl_device)=gpu_input.pad(padding).convolve(gpu_kernel, dims);
  sycl_device.memcpyDeviceToHost(same.data(), d_same, same_bytes);

  VERIFY_IS_EQUAL(same.dimension(0), 3);
  VERIFY_IS_APPROX(same(0), 1.0f);
  VERIFY_IS_APPROX(same(1), 2.5f);
  VERIFY_IS_APPROX(same(2), 4.0f);

  // Emulate FULL mode (as defined in
  // http://docs.scipy.org/doc/numpy/reference/generated/numpy.convolve.html).
  padding[0] = std::make_pair(2, 2);

  Tensor<DataType, 1, DataLayout, IndexType> full(5);
  std::size_t full_bytes = full.size() * sizeof(DataType);
  DataType * d_full =  static_cast<DataType*>(sycl_device.allocate(full_bytes));
  Eigen::TensorMap<Eigen::Tensor<DataType, 1, DataLayout,IndexType> > gpu_full(d_full, full.dimensions());
  gpu_full.device(sycl_device)=gpu_input.pad(padding).convolve(gpu_kernel, dims);
  sycl_device.memcpyDeviceToHost(full.data(), d_full, full_bytes);

  VERIFY_IS_EQUAL(full.dimension(0), 5);
  VERIFY_IS_APPROX(full(0), 0.0f);
  VERIFY_IS_APPROX(full(1), 1.0f);
  VERIFY_IS_APPROX(full(2), 2.5f);
  VERIFY_IS_APPROX(full(3), 4.0f);
  VERIFY_IS_APPROX(full(4), 1.5f);

  sycl_device.deallocate(d_input);
  sycl_device.deallocate(d_kernel);
  sycl_device.deallocate(d_valid);
  sycl_device.deallocate(d_same);
  sycl_device.deallocate(d_full);

}

template <typename DataType, int DataLayout, typename IndexType>
static void test_strides(const Eigen::SyclDevice& sycl_device){

  Eigen::array<IndexType, 1> input_dims = {{13}};
  Eigen::array<IndexType, 1> kernel_dims = {{3}};

  Tensor<DataType, 1, DataLayout, IndexType> input(input_dims);
  Tensor<DataType, 1, DataLayout, IndexType> kernel(kernel_dims);
  Tensor<DataType, 1, DataLayout, IndexType> result(2);

  input.setRandom();
  kernel.setRandom();
  Eigen::array<IndexType, 1> dims;
  dims[0] = 0;

  Eigen::array<IndexType, 1> stride_of_3;
  stride_of_3[0] = 3;
  Eigen::array<IndexType, 1> stride_of_2;
  stride_of_2[0] = 2;

  std::size_t input_bytes = input.size()  * sizeof(DataType);
  std::size_t kernel_bytes = kernel.size() * sizeof(DataType);
  std::size_t result_bytes = result.size() * sizeof(DataType);

  DataType * d_input  = static_cast<DataType*>(sycl_device.allocate(input_bytes));
  DataType * d_kernel  = static_cast<DataType*>(sycl_device.allocate(kernel_bytes));
  DataType * d_result =  static_cast<DataType*>(sycl_device.allocate(result_bytes));

  Eigen::TensorMap<Eigen::Tensor<DataType, 1, DataLayout,IndexType> > gpu_input(d_input, input_dims);
  Eigen::TensorMap<Eigen::Tensor<DataType, 1, DataLayout,IndexType> > gpu_kernel(d_kernel, kernel_dims);
  Eigen::TensorMap<Eigen::Tensor<DataType, 1, DataLayout,IndexType> > gpu_result(d_result, result.dimensions());
  sycl_device.memcpyHostToDevice(d_input, input.data(), input_bytes);
  sycl_device.memcpyHostToDevice(d_kernel, kernel.data(), kernel_bytes);

  gpu_result.device(sycl_device)=gpu_input.stride(stride_of_3).convolve(gpu_kernel, dims).stride(stride_of_2);
  sycl_device.memcpyDeviceToHost(result.data(), d_result, result_bytes);

  VERIFY_IS_EQUAL(result.dimension(0), 2);
  VERIFY_IS_APPROX(result(0), (input(0)*kernel(0) + input(3)*kernel(1) +
                               input(6)*kernel(2)));
  VERIFY_IS_APPROX(result(1), (input(6)*kernel(0) + input(9)*kernel(1) +
                               input(12)*kernel(2)));
}

template <typename Dev_selector> void tensorConvolutionPerDevice(Dev_selector& s){
  QueueInterface queueInterface(s);
  auto sycl_device=Eigen::SyclDevice(&queueInterface);
  test_larg_expr1D<float, RowMajor, int64_t>(sycl_device);
  test_larg_expr1D<float, ColMajor, int64_t>(sycl_device);
  test_larg_expr2D<float, RowMajor, int64_t>(sycl_device);
  test_larg_expr2D<float, ColMajor, int64_t>(sycl_device);
  test_larg_expr3D<float, RowMajor, int64_t>(sycl_device);
  test_larg_expr3D<float, ColMajor, int64_t>(sycl_device);
  test_evals<float, ColMajor, int64_t>(sycl_device);
  test_evals<float, RowMajor, int64_t>(sycl_device);
  test_expr<float, ColMajor, int64_t>(sycl_device);
  test_expr<float, RowMajor, int64_t>(sycl_device);
  test_modes<float, ColMajor, int64_t>(sycl_device);
  test_modes<float, RowMajor, int64_t>(sycl_device);
  test_strides<float, ColMajor, int64_t>(sycl_device);
  test_strides<float, RowMajor, int64_t>(sycl_device);
}

EIGEN_DECLARE_TEST(cxx11_tensor_convolution_sycl) {
  for (const auto& device :Eigen::get_sycl_supported_devices()) {
    CALL_SUBTEST(tensorConvolutionPerDevice(device));
  }
}
