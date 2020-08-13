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

template <typename DataType, int DataLayout, typename IndexType>
static void test_sycl_random_uniform(const Eigen::SyclDevice& sycl_device)
{
  Tensor<DataType, 2,DataLayout, IndexType> out(72,97);
  out.setZero();

  std::size_t out_bytes = out.size() * sizeof(DataType);

  IndexType sizeDim0 = 72;
  IndexType sizeDim1 = 97;

  array<IndexType, 2> tensorRange = {{sizeDim0, sizeDim1}};

  DataType* d_out  = static_cast<DataType*>(sycl_device.allocate(out_bytes));
  TensorMap<Tensor<DataType, 2, DataLayout, IndexType>> gpu_out(d_out, tensorRange);

  gpu_out.device(sycl_device)=gpu_out.random();
  sycl_device.memcpyDeviceToHost(out.data(), d_out,out_bytes);
  for(IndexType i=1; i<sizeDim0; i++)
    for(IndexType j=1; j<sizeDim1; j++)
    {
      VERIFY_IS_NOT_EQUAL(out(i,j), out(i-1,j));
      VERIFY_IS_NOT_EQUAL(out(i,j), out(i,j-1));
      VERIFY_IS_NOT_EQUAL(out(i,j), out(i-1,j-1));    }

  // For now we just check thes code doesn't crash.
  // TODO: come up with a valid test of randomness
  sycl_device.deallocate(d_out);
}

template <typename DataType, int DataLayout, typename IndexType>
void test_sycl_random_normal(const Eigen::SyclDevice& sycl_device)
{
  Tensor<DataType, 2,DataLayout,IndexType> out(72,97);
  out.setZero();
  std::size_t out_bytes = out.size() * sizeof(DataType);

  IndexType sizeDim0 = 72;
  IndexType sizeDim1 = 97;

  array<IndexType, 2> tensorRange = {{sizeDim0, sizeDim1}};

  DataType* d_out  = static_cast<DataType*>(sycl_device.allocate(out_bytes));
  TensorMap<Tensor<DataType, 2, DataLayout, IndexType>> gpu_out(d_out, tensorRange);
  Eigen::internal::NormalRandomGenerator<DataType> gen(true);
  gpu_out.device(sycl_device)=gpu_out.random(gen);
  sycl_device.memcpyDeviceToHost(out.data(), d_out,out_bytes);
  for(IndexType i=1; i<sizeDim0; i++)
    for(IndexType j=1; j<sizeDim1; j++)
    {
      VERIFY_IS_NOT_EQUAL(out(i,j), out(i-1,j));
      VERIFY_IS_NOT_EQUAL(out(i,j), out(i,j-1));
      VERIFY_IS_NOT_EQUAL(out(i,j), out(i-1,j-1));

    }

  // For now we just check thes code doesn't crash.
  // TODO: come up with a valid test of randomness
  sycl_device.deallocate(d_out);
}

template<typename DataType, typename dev_Selector> void sycl_random_test_per_device(dev_Selector s){
  QueueInterface queueInterface(s);
  auto sycl_device = Eigen::SyclDevice(&queueInterface);
  test_sycl_random_uniform<DataType, RowMajor, int64_t>(sycl_device);
  test_sycl_random_uniform<DataType, ColMajor, int64_t>(sycl_device);
  test_sycl_random_normal<DataType, RowMajor, int64_t>(sycl_device);
  test_sycl_random_normal<DataType, ColMajor, int64_t>(sycl_device);

}
EIGEN_DECLARE_TEST(cxx11_tensor_random_sycl)
{
  for (const auto& device :Eigen::get_sycl_supported_devices()) {
    CALL_SUBTEST(sycl_random_test_per_device<float>(device));
#ifdef EIGEN_SYCL_DOUBLE_SUPPORT
    CALL_SUBTEST(sycl_random_test_per_device<double>(device));
#endif
  }
}
