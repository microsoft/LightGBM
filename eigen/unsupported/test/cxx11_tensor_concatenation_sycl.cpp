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

#include "main.h"
#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::Tensor;

template<typename DataType, int DataLayout, typename IndexType>
static void test_simple_concatenation(const Eigen::SyclDevice& sycl_device)
{
  IndexType leftDim1 = 2;
  IndexType leftDim2 = 3;
  IndexType leftDim3 = 1;
  Eigen::array<IndexType, 3> leftRange = {{leftDim1, leftDim2, leftDim3}};
  IndexType rightDim1 = 2;
  IndexType rightDim2 = 3;
  IndexType rightDim3 = 1;
  Eigen::array<IndexType, 3> rightRange = {{rightDim1, rightDim2, rightDim3}};

  //IndexType concatDim1 = 3;
//	IndexType concatDim2 = 3;
//	IndexType concatDim3 = 1;
  //Eigen::array<IndexType, 3> concatRange = {{concatDim1, concatDim2, concatDim3}};

  Tensor<DataType, 3, DataLayout, IndexType> left(leftRange);
  Tensor<DataType, 3, DataLayout, IndexType> right(rightRange);
  left.setRandom();
  right.setRandom();

  DataType * gpu_in1_data  = static_cast<DataType*>(sycl_device.allocate(left.dimensions().TotalSize()*sizeof(DataType)));
  DataType * gpu_in2_data  = static_cast<DataType*>(sycl_device.allocate(right.dimensions().TotalSize()*sizeof(DataType)));

  Eigen::TensorMap<Eigen::Tensor<DataType, 3, DataLayout, IndexType>> gpu_in1(gpu_in1_data, leftRange);
  Eigen::TensorMap<Eigen::Tensor<DataType, 3, DataLayout, IndexType>> gpu_in2(gpu_in2_data, rightRange);
  sycl_device.memcpyHostToDevice(gpu_in1_data, left.data(),(left.dimensions().TotalSize())*sizeof(DataType));
  sycl_device.memcpyHostToDevice(gpu_in2_data, right.data(),(right.dimensions().TotalSize())*sizeof(DataType));
  ///
  Tensor<DataType, 3, DataLayout, IndexType> concatenation1(leftDim1+rightDim1, leftDim2, leftDim3);
  DataType * gpu_out_data1 =  static_cast<DataType*>(sycl_device.allocate(concatenation1.dimensions().TotalSize()*sizeof(DataType)));
  Eigen::TensorMap<Eigen::Tensor<DataType, 3, DataLayout, IndexType>> gpu_out1(gpu_out_data1, concatenation1.dimensions());

  //concatenation = left.concatenate(right, 0);
  gpu_out1.device(sycl_device) =gpu_in1.concatenate(gpu_in2, 0);
  sycl_device.memcpyDeviceToHost(concatenation1.data(), gpu_out_data1,(concatenation1.dimensions().TotalSize())*sizeof(DataType));

  VERIFY_IS_EQUAL(concatenation1.dimension(0), 4);
  VERIFY_IS_EQUAL(concatenation1.dimension(1), 3);
  VERIFY_IS_EQUAL(concatenation1.dimension(2), 1);
  for (IndexType j = 0; j < 3; ++j) {
    for (IndexType i = 0; i < 2; ++i) {
      VERIFY_IS_EQUAL(concatenation1(i, j, 0), left(i, j, 0));
    }
    for (IndexType i = 2; i < 4; ++i) {
      VERIFY_IS_EQUAL(concatenation1(i, j, 0), right(i - 2, j, 0));
    }
  }

  sycl_device.deallocate(gpu_out_data1);
  Tensor<DataType, 3, DataLayout, IndexType> concatenation2(leftDim1, leftDim2 +rightDim2, leftDim3);
  DataType * gpu_out_data2 =  static_cast<DataType*>(sycl_device.allocate(concatenation2.dimensions().TotalSize()*sizeof(DataType)));
  Eigen::TensorMap<Eigen::Tensor<DataType, 3, DataLayout, IndexType>> gpu_out2(gpu_out_data2, concatenation2.dimensions());
  gpu_out2.device(sycl_device) =gpu_in1.concatenate(gpu_in2, 1);
  sycl_device.memcpyDeviceToHost(concatenation2.data(), gpu_out_data2,(concatenation2.dimensions().TotalSize())*sizeof(DataType));

  //concatenation = left.concatenate(right, 1);
  VERIFY_IS_EQUAL(concatenation2.dimension(0), 2);
  VERIFY_IS_EQUAL(concatenation2.dimension(1), 6);
  VERIFY_IS_EQUAL(concatenation2.dimension(2), 1);
  for (IndexType i = 0; i < 2; ++i) {
    for (IndexType j = 0; j < 3; ++j) {
      VERIFY_IS_EQUAL(concatenation2(i, j, 0), left(i, j, 0));
    }
    for (IndexType j = 3; j < 6; ++j) {
      VERIFY_IS_EQUAL(concatenation2(i, j, 0), right(i, j - 3, 0));
    }
  }
  sycl_device.deallocate(gpu_out_data2);
  Tensor<DataType, 3, DataLayout, IndexType> concatenation3(leftDim1, leftDim2, leftDim3+rightDim3);
  DataType * gpu_out_data3 =  static_cast<DataType*>(sycl_device.allocate(concatenation3.dimensions().TotalSize()*sizeof(DataType)));
  Eigen::TensorMap<Eigen::Tensor<DataType, 3, DataLayout, IndexType>> gpu_out3(gpu_out_data3, concatenation3.dimensions());
  gpu_out3.device(sycl_device) =gpu_in1.concatenate(gpu_in2, 2);
  sycl_device.memcpyDeviceToHost(concatenation3.data(), gpu_out_data3,(concatenation3.dimensions().TotalSize())*sizeof(DataType));

  //concatenation = left.concatenate(right, 2);
  VERIFY_IS_EQUAL(concatenation3.dimension(0), 2);
  VERIFY_IS_EQUAL(concatenation3.dimension(1), 3);
  VERIFY_IS_EQUAL(concatenation3.dimension(2), 2);
  for (IndexType i = 0; i < 2; ++i) {
    for (IndexType j = 0; j < 3; ++j) {
      VERIFY_IS_EQUAL(concatenation3(i, j, 0), left(i, j, 0));
      VERIFY_IS_EQUAL(concatenation3(i, j, 1), right(i, j, 0));
    }
  }
  sycl_device.deallocate(gpu_out_data3);
  sycl_device.deallocate(gpu_in1_data);
  sycl_device.deallocate(gpu_in2_data);
}
template<typename DataType, int DataLayout, typename IndexType>
static void test_concatenation_as_lvalue(const Eigen::SyclDevice& sycl_device)
{

  IndexType leftDim1 = 2;
  IndexType leftDim2 = 3;
  Eigen::array<IndexType, 2> leftRange = {{leftDim1, leftDim2}};

  IndexType rightDim1 = 2;
  IndexType rightDim2 = 3;
  Eigen::array<IndexType, 2> rightRange = {{rightDim1, rightDim2}};

  IndexType concatDim1 = 4;
  IndexType concatDim2 = 3;
  Eigen::array<IndexType, 2> resRange = {{concatDim1, concatDim2}};

  Tensor<DataType, 2, DataLayout, IndexType> left(leftRange);
  Tensor<DataType, 2, DataLayout, IndexType> right(rightRange);
  Tensor<DataType, 2, DataLayout, IndexType> result(resRange);

  left.setRandom();
  right.setRandom();
  result.setRandom();

  DataType * gpu_in1_data  = static_cast<DataType*>(sycl_device.allocate(left.dimensions().TotalSize()*sizeof(DataType)));
  DataType * gpu_in2_data  = static_cast<DataType*>(sycl_device.allocate(right.dimensions().TotalSize()*sizeof(DataType)));
  DataType * gpu_out_data =  static_cast<DataType*>(sycl_device.allocate(result.dimensions().TotalSize()*sizeof(DataType)));


  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType>> gpu_in1(gpu_in1_data, leftRange);
  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType>> gpu_in2(gpu_in2_data, rightRange);
  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType>> gpu_out(gpu_out_data, resRange);

  sycl_device.memcpyHostToDevice(gpu_in1_data, left.data(),(left.dimensions().TotalSize())*sizeof(DataType));
  sycl_device.memcpyHostToDevice(gpu_in2_data, right.data(),(right.dimensions().TotalSize())*sizeof(DataType));
  sycl_device.memcpyHostToDevice(gpu_out_data, result.data(),(result.dimensions().TotalSize())*sizeof(DataType));

//  t1.concatenate(t2, 0) = result;
 gpu_in1.concatenate(gpu_in2, 0).device(sycl_device) =gpu_out;
 sycl_device.memcpyDeviceToHost(left.data(), gpu_in1_data,(left.dimensions().TotalSize())*sizeof(DataType));
 sycl_device.memcpyDeviceToHost(right.data(), gpu_in2_data,(right.dimensions().TotalSize())*sizeof(DataType));

  for (IndexType i = 0; i < 2; ++i) {
    for (IndexType j = 0; j < 3; ++j) {
      VERIFY_IS_EQUAL(left(i, j), result(i, j));
      VERIFY_IS_EQUAL(right(i, j), result(i+2, j));
    }
  }
  sycl_device.deallocate(gpu_in1_data);
  sycl_device.deallocate(gpu_in2_data);
  sycl_device.deallocate(gpu_out_data);
}


template <typename DataType, typename Dev_selector> void tensorConcat_perDevice(Dev_selector s){
  QueueInterface queueInterface(s);
  auto sycl_device = Eigen::SyclDevice(&queueInterface);
  test_simple_concatenation<DataType, RowMajor, int64_t>(sycl_device);
  test_simple_concatenation<DataType, ColMajor, int64_t>(sycl_device);
  test_concatenation_as_lvalue<DataType, ColMajor, int64_t>(sycl_device);
}
EIGEN_DECLARE_TEST(cxx11_tensor_concatenation_sycl) {
  for (const auto& device :Eigen::get_sycl_supported_devices()) {
    CALL_SUBTEST(tensorConcat_perDevice<float>(device));
  }
}
