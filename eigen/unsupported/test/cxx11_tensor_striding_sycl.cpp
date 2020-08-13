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

using Eigen::array;
using Eigen::SyclDevice;
using Eigen::Tensor;
using Eigen::TensorMap;


template <typename DataType, int DataLayout, typename IndexType>
static void test_simple_striding(const Eigen::SyclDevice& sycl_device)
{

  Eigen::array<IndexType, 4> tensor_dims = {{2,3,5,7}};
  Eigen::array<IndexType, 4> stride_dims = {{1,1,3,3}};


  Tensor<DataType, 4, DataLayout, IndexType> tensor(tensor_dims);
  Tensor<DataType, 4, DataLayout,IndexType> no_stride(tensor_dims);
  Tensor<DataType, 4, DataLayout,IndexType> stride(stride_dims);


  std::size_t tensor_bytes = tensor.size()  * sizeof(DataType);
  std::size_t no_stride_bytes = no_stride.size() * sizeof(DataType);
  std::size_t stride_bytes = stride.size() * sizeof(DataType);
  DataType * d_tensor = static_cast<DataType*>(sycl_device.allocate(tensor_bytes));
  DataType * d_no_stride = static_cast<DataType*>(sycl_device.allocate(no_stride_bytes));
  DataType * d_stride = static_cast<DataType*>(sycl_device.allocate(stride_bytes));

  Eigen::TensorMap<Eigen::Tensor<DataType, 4, DataLayout, IndexType> > gpu_tensor(d_tensor, tensor_dims);
  Eigen::TensorMap<Eigen::Tensor<DataType, 4, DataLayout, IndexType> > gpu_no_stride(d_no_stride, tensor_dims);
  Eigen::TensorMap<Eigen::Tensor<DataType, 4, DataLayout, IndexType> > gpu_stride(d_stride, stride_dims);


  tensor.setRandom();
  array<IndexType, 4> strides;
  strides[0] = 1;
  strides[1] = 1;
  strides[2] = 1;
  strides[3] = 1;
  sycl_device.memcpyHostToDevice(d_tensor, tensor.data(), tensor_bytes);
  gpu_no_stride.device(sycl_device)=gpu_tensor.stride(strides);
  sycl_device.memcpyDeviceToHost(no_stride.data(), d_no_stride, no_stride_bytes);

  //no_stride = tensor.stride(strides);

  VERIFY_IS_EQUAL(no_stride.dimension(0), 2);
  VERIFY_IS_EQUAL(no_stride.dimension(1), 3);
  VERIFY_IS_EQUAL(no_stride.dimension(2), 5);
  VERIFY_IS_EQUAL(no_stride.dimension(3), 7);

  for (IndexType i = 0; i < 2; ++i) {
    for (IndexType j = 0; j < 3; ++j) {
      for (IndexType k = 0; k < 5; ++k) {
        for (IndexType l = 0; l < 7; ++l) {
          VERIFY_IS_EQUAL(tensor(i,j,k,l), no_stride(i,j,k,l));
        }
      }
    }
  }

  strides[0] = 2;
  strides[1] = 4;
  strides[2] = 2;
  strides[3] = 3;
//Tensor<float, 4, DataLayout> stride;
//  stride = tensor.stride(strides);

  gpu_stride.device(sycl_device)=gpu_tensor.stride(strides);
  sycl_device.memcpyDeviceToHost(stride.data(), d_stride, stride_bytes);

  VERIFY_IS_EQUAL(stride.dimension(0), 1);
  VERIFY_IS_EQUAL(stride.dimension(1), 1);
  VERIFY_IS_EQUAL(stride.dimension(2), 3);
  VERIFY_IS_EQUAL(stride.dimension(3), 3);

  for (IndexType i = 0; i < 1; ++i) {
    for (IndexType j = 0; j < 1; ++j) {
      for (IndexType k = 0; k < 3; ++k) {
        for (IndexType l = 0; l < 3; ++l) {
          VERIFY_IS_EQUAL(tensor(2*i,4*j,2*k,3*l), stride(i,j,k,l));
        }
      }
    }
  }

  sycl_device.deallocate(d_tensor);
  sycl_device.deallocate(d_no_stride);
  sycl_device.deallocate(d_stride);
}

template <typename DataType, int DataLayout, typename IndexType>
static void test_striding_as_lvalue(const Eigen::SyclDevice& sycl_device)
{

  Eigen::array<IndexType, 4> tensor_dims = {{2,3,5,7}};
  Eigen::array<IndexType, 4> stride_dims = {{3,12,10,21}};


  Tensor<DataType, 4, DataLayout, IndexType> tensor(tensor_dims);
  Tensor<DataType, 4, DataLayout,IndexType> no_stride(stride_dims);
  Tensor<DataType, 4, DataLayout,IndexType> stride(stride_dims);


  std::size_t tensor_bytes = tensor.size()  * sizeof(DataType);
  std::size_t no_stride_bytes = no_stride.size() * sizeof(DataType);
  std::size_t stride_bytes = stride.size() * sizeof(DataType);

  DataType * d_tensor = static_cast<DataType*>(sycl_device.allocate(tensor_bytes));
  DataType * d_no_stride = static_cast<DataType*>(sycl_device.allocate(no_stride_bytes));
  DataType * d_stride = static_cast<DataType*>(sycl_device.allocate(stride_bytes));

  Eigen::TensorMap<Eigen::Tensor<DataType, 4, DataLayout, IndexType> > gpu_tensor(d_tensor, tensor_dims);
  Eigen::TensorMap<Eigen::Tensor<DataType, 4, DataLayout, IndexType> > gpu_no_stride(d_no_stride, stride_dims);
  Eigen::TensorMap<Eigen::Tensor<DataType, 4, DataLayout, IndexType> > gpu_stride(d_stride, stride_dims);

  //Tensor<float, 4, DataLayout> tensor(2,3,5,7);
  tensor.setRandom();
  array<IndexType, 4> strides;
  strides[0] = 2;
  strides[1] = 4;
  strides[2] = 2;
  strides[3] = 3;

//  Tensor<float, 4, DataLayout> result(3, 12, 10, 21);
//  result.stride(strides) = tensor;
  sycl_device.memcpyHostToDevice(d_tensor, tensor.data(), tensor_bytes);
  gpu_stride.stride(strides).device(sycl_device)=gpu_tensor;
  sycl_device.memcpyDeviceToHost(stride.data(), d_stride, stride_bytes);

  for (IndexType i = 0; i < 2; ++i) {
    for (IndexType j = 0; j < 3; ++j) {
      for (IndexType k = 0; k < 5; ++k) {
        for (IndexType l = 0; l < 7; ++l) {
          VERIFY_IS_EQUAL(tensor(i,j,k,l), stride(2*i,4*j,2*k,3*l));
        }
      }
    }
  }

  array<IndexType, 4> no_strides;
  no_strides[0] = 1;
  no_strides[1] = 1;
  no_strides[2] = 1;
  no_strides[3] = 1;
//  Tensor<float, 4, DataLayout> result2(3, 12, 10, 21);
//  result2.stride(strides) = tensor.stride(no_strides);

  gpu_no_stride.stride(strides).device(sycl_device)=gpu_tensor.stride(no_strides);
  sycl_device.memcpyDeviceToHost(no_stride.data(), d_no_stride, no_stride_bytes);

  for (IndexType i = 0; i < 2; ++i) {
    for (IndexType j = 0; j < 3; ++j) {
      for (IndexType k = 0; k < 5; ++k) {
        for (IndexType l = 0; l < 7; ++l) {
          VERIFY_IS_EQUAL(tensor(i,j,k,l), no_stride(2*i,4*j,2*k,3*l));
        }
      }
    }
  }
  sycl_device.deallocate(d_tensor);
  sycl_device.deallocate(d_no_stride);
  sycl_device.deallocate(d_stride);
}


template <typename Dev_selector> void tensorStridingPerDevice(Dev_selector& s){
  QueueInterface queueInterface(s);
  auto sycl_device=Eigen::SyclDevice(&queueInterface);
  test_simple_striding<float, ColMajor, int64_t>(sycl_device);
  test_simple_striding<float, RowMajor, int64_t>(sycl_device);
  test_striding_as_lvalue<float, ColMajor, int64_t>(sycl_device);
  test_striding_as_lvalue<float, RowMajor, int64_t>(sycl_device);
}

EIGEN_DECLARE_TEST(cxx11_tensor_striding_sycl) {
  for (const auto& device :Eigen::get_sycl_supported_devices()) {
    CALL_SUBTEST(tensorStridingPerDevice(device));
  }
}
