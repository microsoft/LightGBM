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

using Eigen::array;
using Eigen::SyclDevice;
using Eigen::Tensor;
using Eigen::TensorMap;

template <typename DataType, int DataLayout, typename IndexType>
static void test_broadcast_sycl_fixed(const Eigen::SyclDevice &sycl_device){

  // BROADCAST test:
  IndexType inDim1=2;
  IndexType inDim2=3;
  IndexType inDim3=5;
  IndexType inDim4=7;
  IndexType bDim1=2;
  IndexType bDim2=3;
  IndexType bDim3=1;
  IndexType bDim4=4;
  array<IndexType, 4> in_range   = {{inDim1, inDim2, inDim3, inDim4}};
  array<IndexType, 4> broadcasts = {{bDim1, bDim2, bDim3, bDim4}};
  array<IndexType, 4> out_range;  // = in_range * broadcasts
  for (size_t i = 0; i < out_range.size(); ++i)
    out_range[i] = in_range[i] * broadcasts[i];

  Tensor<DataType, 4, DataLayout, IndexType>  input(in_range);
  Tensor<DataType, 4, DataLayout, IndexType> out(out_range);

  for (size_t i = 0; i < in_range.size(); ++i)
    VERIFY_IS_EQUAL(out.dimension(i), out_range[i]);


  for (IndexType i = 0; i < input.size(); ++i)
    input(i) = static_cast<DataType>(i);

  DataType * gpu_in_data  = static_cast<DataType*>(sycl_device.allocate(input.dimensions().TotalSize()*sizeof(DataType)));
  DataType * gpu_out_data  = static_cast<DataType*>(sycl_device.allocate(out.dimensions().TotalSize()*sizeof(DataType)));

  TensorMap<TensorFixedSize<DataType, Sizes<2, 3, 5, 7>, DataLayout, IndexType>> gpu_in(gpu_in_data, in_range);
  TensorMap<Tensor<DataType, 4, DataLayout, IndexType>> gpu_out(gpu_out_data, out_range);
  sycl_device.memcpyHostToDevice(gpu_in_data, input.data(),(input.dimensions().TotalSize())*sizeof(DataType));
  gpu_out.device(sycl_device) = gpu_in.broadcast(broadcasts);
  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.dimensions().TotalSize())*sizeof(DataType));

  for (IndexType i = 0; i < inDim1*bDim1; ++i) {
    for (IndexType j = 0; j < inDim2*bDim2; ++j) {
      for (IndexType k = 0; k < inDim3*bDim3; ++k) {
        for (IndexType l = 0; l < inDim4*bDim4; ++l) {
          VERIFY_IS_APPROX(input(i%2,j%3,k%5,l%7), out(i,j,k,l));
        }
      }
    }
  }
  printf("Broadcast Test with fixed size Passed\n");
  sycl_device.deallocate(gpu_in_data);
  sycl_device.deallocate(gpu_out_data);
}

template <typename DataType, int DataLayout, typename IndexType>
static void test_broadcast_sycl(const Eigen::SyclDevice &sycl_device){

  // BROADCAST test:
  IndexType inDim1=2;
  IndexType inDim2=3;
  IndexType inDim3=5;
  IndexType inDim4=7;
  IndexType bDim1=2;
  IndexType bDim2=3;
  IndexType bDim3=1;
  IndexType bDim4=4;
  array<IndexType, 4> in_range   = {{inDim1, inDim2, inDim3, inDim4}};
  array<IndexType, 4> broadcasts = {{bDim1, bDim2, bDim3, bDim4}};
  array<IndexType, 4> out_range;  // = in_range * broadcasts
  for (size_t i = 0; i < out_range.size(); ++i)
    out_range[i] = in_range[i] * broadcasts[i];

  Tensor<DataType, 4, DataLayout, IndexType>  input(in_range);
  Tensor<DataType, 4, DataLayout, IndexType> out(out_range);

  for (size_t i = 0; i < in_range.size(); ++i)
    VERIFY_IS_EQUAL(out.dimension(i), out_range[i]);


  for (IndexType i = 0; i < input.size(); ++i)
    input(i) = static_cast<DataType>(i);

  DataType * gpu_in_data  = static_cast<DataType*>(sycl_device.allocate(input.dimensions().TotalSize()*sizeof(DataType)));
  DataType * gpu_out_data  = static_cast<DataType*>(sycl_device.allocate(out.dimensions().TotalSize()*sizeof(DataType)));

  TensorMap<Tensor<DataType, 4, DataLayout, IndexType>>  gpu_in(gpu_in_data, in_range);
  TensorMap<Tensor<DataType, 4, DataLayout, IndexType>> gpu_out(gpu_out_data, out_range);
  sycl_device.memcpyHostToDevice(gpu_in_data, input.data(),(input.dimensions().TotalSize())*sizeof(DataType));
  gpu_out.device(sycl_device) = gpu_in.broadcast(broadcasts);
  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.dimensions().TotalSize())*sizeof(DataType));

  for (IndexType i = 0; i < inDim1*bDim1; ++i) {
    for (IndexType j = 0; j < inDim2*bDim2; ++j) {
      for (IndexType k = 0; k < inDim3*bDim3; ++k) {
        for (IndexType l = 0; l < inDim4*bDim4; ++l) {
          VERIFY_IS_APPROX(input(i%inDim1,j%inDim2,k%inDim3,l%inDim4), out(i,j,k,l));
        }
      }
    }
  }
  printf("Broadcast Test Passed\n");
  sycl_device.deallocate(gpu_in_data);
  sycl_device.deallocate(gpu_out_data);
}

template<typename DataType> void sycl_broadcast_test_per_device(const cl::sycl::device& d){
  std::cout << "Running on " << d.template get_info<cl::sycl::info::device::name>() << std::endl;
  QueueInterface queueInterface(d);
  auto sycl_device = Eigen::SyclDevice(&queueInterface);
  test_broadcast_sycl<DataType, RowMajor, int64_t>(sycl_device);
  test_broadcast_sycl<DataType, ColMajor, int64_t>(sycl_device);
  test_broadcast_sycl_fixed<DataType, RowMajor, int64_t>(sycl_device);
  test_broadcast_sycl_fixed<DataType, ColMajor, int64_t>(sycl_device);
}

EIGEN_DECLARE_TEST(cxx11_tensor_broadcast_sycl) {
  for (const auto& device :Eigen::get_sycl_supported_devices()) {
    CALL_SUBTEST(sycl_broadcast_test_per_device<float>(device));
  }
}
