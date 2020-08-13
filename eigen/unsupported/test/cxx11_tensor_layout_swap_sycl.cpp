// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016
// Mehdi Goli    Codeplay Software Ltd.
// Ralph Potter  Codeplay Software Ltd.
// Luke Iwanski  Codeplay Software Ltd.
// Contact: <eigen@codeplay.com>
// Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_NO_COMPLEX

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int64_t
#define EIGEN_USE_SYCL

#include "main.h"

#include <Eigen/CXX11/Tensor>

using Eigen::Tensor;

template <typename DataType, typename IndexType>
static void test_simple_swap_sycl(const Eigen::SyclDevice& sycl_device)
{
  IndexType sizeDim1 = 2;
  IndexType sizeDim2 = 3;
  IndexType sizeDim3 = 7;
  array<IndexType, 3> tensorColRange = {{sizeDim1, sizeDim2, sizeDim3}};
  array<IndexType, 3> tensorRowRange = {{sizeDim3, sizeDim2, sizeDim1}};


  Tensor<DataType, 3, ColMajor, IndexType> tensor1(tensorColRange);
  Tensor<DataType, 3, RowMajor, IndexType> tensor2(tensorRowRange);
  tensor1.setRandom();

  DataType* gpu_data1  = static_cast<DataType*>(sycl_device.allocate(tensor1.size()*sizeof(DataType)));
  DataType* gpu_data2  = static_cast<DataType*>(sycl_device.allocate(tensor2.size()*sizeof(DataType)));
  TensorMap<Tensor<DataType, 3, ColMajor, IndexType>> gpu1(gpu_data1, tensorColRange);
  TensorMap<Tensor<DataType, 3, RowMajor, IndexType>> gpu2(gpu_data2, tensorRowRange);

  sycl_device.memcpyHostToDevice(gpu_data1, tensor1.data(),(tensor1.size())*sizeof(DataType));
  gpu2.device(sycl_device)=gpu1.swap_layout();
  sycl_device.memcpyDeviceToHost(tensor2.data(), gpu_data2,(tensor2.size())*sizeof(DataType));


//  Tensor<float, 3, ColMajor> tensor(2,3,7);
  //tensor.setRandom();

//  Tensor<float, 3, RowMajor> tensor2 = tensor.swap_layout();
  VERIFY_IS_EQUAL(tensor1.dimension(0), tensor2.dimension(2));
  VERIFY_IS_EQUAL(tensor1.dimension(1), tensor2.dimension(1));
  VERIFY_IS_EQUAL(tensor1.dimension(2), tensor2.dimension(0));

  for (IndexType i = 0; i < 2; ++i) {
    for (IndexType j = 0; j < 3; ++j) {
      for (IndexType k = 0; k < 7; ++k) {
        VERIFY_IS_EQUAL(tensor1(i,j,k), tensor2(k,j,i));
      }
    }
  }
  sycl_device.deallocate(gpu_data1);
  sycl_device.deallocate(gpu_data2);
}

template <typename DataType, typename IndexType>
static void test_swap_as_lvalue_sycl(const Eigen::SyclDevice& sycl_device)
{

  IndexType sizeDim1 = 2;
  IndexType sizeDim2 = 3;
  IndexType sizeDim3 = 7;
  array<IndexType, 3> tensorColRange = {{sizeDim1, sizeDim2, sizeDim3}};
  array<IndexType, 3> tensorRowRange = {{sizeDim3, sizeDim2, sizeDim1}};

  Tensor<DataType, 3, ColMajor, IndexType> tensor1(tensorColRange);
  Tensor<DataType, 3, RowMajor, IndexType> tensor2(tensorRowRange);
  tensor1.setRandom();

  DataType* gpu_data1  = static_cast<DataType*>(sycl_device.allocate(tensor1.size()*sizeof(DataType)));
  DataType* gpu_data2  = static_cast<DataType*>(sycl_device.allocate(tensor2.size()*sizeof(DataType)));
  TensorMap<Tensor<DataType, 3, ColMajor, IndexType>> gpu1(gpu_data1, tensorColRange);
  TensorMap<Tensor<DataType, 3, RowMajor, IndexType>> gpu2(gpu_data2, tensorRowRange);

  sycl_device.memcpyHostToDevice(gpu_data1, tensor1.data(),(tensor1.size())*sizeof(DataType));
  gpu2.swap_layout().device(sycl_device)=gpu1;
  sycl_device.memcpyDeviceToHost(tensor2.data(), gpu_data2,(tensor2.size())*sizeof(DataType));


//  Tensor<float, 3, ColMajor> tensor(2,3,7);
//  tensor.setRandom();

  //Tensor<float, 3, RowMajor> tensor2(7,3,2);
//  tensor2.swap_layout() = tensor;
  VERIFY_IS_EQUAL(tensor1.dimension(0), tensor2.dimension(2));
  VERIFY_IS_EQUAL(tensor1.dimension(1), tensor2.dimension(1));
  VERIFY_IS_EQUAL(tensor1.dimension(2), tensor2.dimension(0));

  for (IndexType i = 0; i < 2; ++i) {
    for (IndexType j = 0; j < 3; ++j) {
      for (IndexType k = 0; k < 7; ++k) {
        VERIFY_IS_EQUAL(tensor1(i,j,k), tensor2(k,j,i));
      }
    }
  }
  sycl_device.deallocate(gpu_data1);
  sycl_device.deallocate(gpu_data2);
}


template<typename DataType, typename dev_Selector> void sycl_tensor_layout_swap_test_per_device(dev_Selector s){
  QueueInterface queueInterface(s);
  auto sycl_device = Eigen::SyclDevice(&queueInterface);
  test_simple_swap_sycl<DataType, int64_t>(sycl_device);
  test_swap_as_lvalue_sycl<DataType, int64_t>(sycl_device);
}
EIGEN_DECLARE_TEST(cxx11_tensor_layout_swap_sycl)
{
  for (const auto& device :Eigen::get_sycl_supported_devices()) {
    CALL_SUBTEST(sycl_tensor_layout_swap_test_per_device<float>(device));
  }
}
