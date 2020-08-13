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
#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::array;
using Eigen::SyclDevice;
using Eigen::Tensor;
using Eigen::TensorMap;

template <typename DataType, int DataLayout, typename IndexType>
void test_sycl_mem_transfers(const Eigen::SyclDevice &sycl_device) {
  IndexType sizeDim1 = 5;
  IndexType sizeDim2 = 5;
  IndexType sizeDim3 = 1;
  array<IndexType, 3> tensorRange = {{sizeDim1, sizeDim2, sizeDim3}};
  Tensor<DataType, 3, DataLayout, IndexType> in1(tensorRange);
  Tensor<DataType, 3, DataLayout, IndexType> out1(tensorRange);
  Tensor<DataType, 3, DataLayout, IndexType> out2(tensorRange);
  Tensor<DataType, 3, DataLayout, IndexType> out3(tensorRange);

  in1 = in1.random();

  DataType* gpu_data1  = static_cast<DataType*>(sycl_device.allocate(in1.size()*sizeof(DataType)));
  DataType* gpu_data2  = static_cast<DataType*>(sycl_device.allocate(out1.size()*sizeof(DataType)));

  TensorMap<Tensor<DataType, 3, DataLayout, IndexType>> gpu1(gpu_data1, tensorRange);
  TensorMap<Tensor<DataType, 3, DataLayout, IndexType>> gpu2(gpu_data2, tensorRange);

  sycl_device.memcpyHostToDevice(gpu_data1, in1.data(),(in1.size())*sizeof(DataType));
  sycl_device.memcpyHostToDevice(gpu_data2, in1.data(),(in1.size())*sizeof(DataType));
  gpu1.device(sycl_device) = gpu1 * 3.14f;
  gpu2.device(sycl_device) = gpu2 * 2.7f;
  sycl_device.memcpyDeviceToHost(out1.data(), gpu_data1,(out1.size())*sizeof(DataType));
  sycl_device.memcpyDeviceToHost(out2.data(), gpu_data1,(out2.size())*sizeof(DataType));
  sycl_device.memcpyDeviceToHost(out3.data(), gpu_data2,(out3.size())*sizeof(DataType));
  sycl_device.synchronize();

  for (IndexType i = 0; i < in1.size(); ++i) {
  //  std::cout << "SYCL DATA : " << out1(i) << "  vs  CPU DATA : " << in1(i) * 3.14f << "\n";
    VERIFY_IS_APPROX(out1(i), in1(i) * 3.14f);
    VERIFY_IS_APPROX(out2(i), in1(i) * 3.14f);
    VERIFY_IS_APPROX(out3(i), in1(i) * 2.7f);
  }

  sycl_device.deallocate(gpu_data1);
  sycl_device.deallocate(gpu_data2);
}

template <typename DataType, int DataLayout, typename IndexType>
void test_sycl_mem_sync(const Eigen::SyclDevice &sycl_device) {
  IndexType size = 20;
  array<IndexType, 1> tensorRange = {{size}};
  Tensor<DataType, 1, DataLayout, IndexType> in1(tensorRange);
  Tensor<DataType, 1, DataLayout, IndexType> in2(tensorRange);
  Tensor<DataType, 1, DataLayout, IndexType> out(tensorRange);

  in1 = in1.random();
  in2 = in1;

  DataType* gpu_data  = static_cast<DataType*>(sycl_device.allocate(in1.size()*sizeof(DataType)));

  TensorMap<Tensor<DataType, 1, DataLayout, IndexType>> gpu1(gpu_data, tensorRange);
  sycl_device.memcpyHostToDevice(gpu_data, in1.data(),(in1.size())*sizeof(DataType));
  sycl_device.synchronize();
  in1.setZero();

  sycl_device.memcpyDeviceToHost(out.data(), gpu_data, out.size()*sizeof(DataType));
  sycl_device.synchronize();

  for (IndexType i = 0; i < in1.size(); ++i) {
    VERIFY_IS_APPROX(out(i), in2(i));
  }

  sycl_device.deallocate(gpu_data);
}

template <typename DataType, int DataLayout, typename IndexType>
void test_sycl_mem_sync_offsets(const Eigen::SyclDevice &sycl_device) {
  using tensor_type = Tensor<DataType, 1, DataLayout, IndexType>;
  IndexType full_size = 32;
  IndexType half_size = full_size / 2;
  array<IndexType, 1> tensorRange = {{full_size}};
  tensor_type in1(tensorRange);
  tensor_type out(tensorRange);

  DataType* gpu_data  = static_cast<DataType*>(sycl_device.allocate(full_size * sizeof(DataType)));
  TensorMap<tensor_type> gpu1(gpu_data, tensorRange);

  in1 = in1.random();
  // Copy all data to device, then permute on copy back to host
  sycl_device.memcpyHostToDevice(gpu_data, in1.data(), full_size * sizeof(DataType));
  sycl_device.memcpyDeviceToHost(out.data(), gpu_data + half_size, half_size * sizeof(DataType));
  sycl_device.memcpyDeviceToHost(out.data() + half_size, gpu_data, half_size * sizeof(DataType));

  for (IndexType i = 0; i < half_size; ++i) {
    VERIFY_IS_APPROX(out(i), in1(i + half_size));
    VERIFY_IS_APPROX(out(i + half_size), in1(i));
  }

  in1 = in1.random();
  out.setZero();
  // Permute copies to device, then copy all back to host
  sycl_device.memcpyHostToDevice(gpu_data + half_size, in1.data(), half_size * sizeof(DataType));
  sycl_device.memcpyHostToDevice(gpu_data, in1.data() + half_size, half_size * sizeof(DataType));
  sycl_device.memcpyDeviceToHost(out.data(), gpu_data, full_size * sizeof(DataType));

  for (IndexType i = 0; i < half_size; ++i) {
    VERIFY_IS_APPROX(out(i), in1(i + half_size));
    VERIFY_IS_APPROX(out(i + half_size), in1(i));
  }

  in1 = in1.random();
  out.setZero();
  DataType* gpu_data_out  = static_cast<DataType*>(sycl_device.allocate(full_size * sizeof(DataType)));
  TensorMap<tensor_type> gpu2(gpu_data_out, tensorRange);
  // Copy all to device, permute copies on device, then copy all back to host
  sycl_device.memcpyHostToDevice(gpu_data, in1.data(), full_size * sizeof(DataType));
  sycl_device.memcpy(gpu_data_out + half_size, gpu_data, half_size * sizeof(DataType));
  sycl_device.memcpy(gpu_data_out, gpu_data + half_size, half_size * sizeof(DataType));
  sycl_device.memcpyDeviceToHost(out.data(), gpu_data_out, full_size * sizeof(DataType));

  for (IndexType i = 0; i < half_size; ++i) {
    VERIFY_IS_APPROX(out(i), in1(i + half_size));
    VERIFY_IS_APPROX(out(i + half_size), in1(i));
  }

  sycl_device.deallocate(gpu_data_out);
  sycl_device.deallocate(gpu_data);
}

template <typename DataType, int DataLayout, typename IndexType>
void test_sycl_memset_offsets(const Eigen::SyclDevice &sycl_device) {
  using tensor_type = Tensor<DataType, 1, DataLayout, IndexType>;
  IndexType full_size = 32;
  IndexType half_size = full_size / 2;
  array<IndexType, 1> tensorRange = {{full_size}};
  tensor_type cpu_out(tensorRange);
  tensor_type out(tensorRange);

  cpu_out.setZero();

  std::memset(cpu_out.data(), 0, half_size * sizeof(DataType));
  std::memset(cpu_out.data() + half_size, 1, half_size * sizeof(DataType));

  DataType* gpu_data  = static_cast<DataType*>(sycl_device.allocate(full_size * sizeof(DataType)));
  TensorMap<tensor_type> gpu1(gpu_data, tensorRange);

  sycl_device.memset(gpu_data, 0, half_size * sizeof(DataType));
  sycl_device.memset(gpu_data + half_size, 1, half_size * sizeof(DataType));
  sycl_device.memcpyDeviceToHost(out.data(), gpu_data, full_size * sizeof(DataType));

  for (IndexType i = 0; i < full_size; ++i) {
    VERIFY_IS_APPROX(out(i), cpu_out(i));
  }

  sycl_device.deallocate(gpu_data);
}

template <typename DataType, int DataLayout, typename IndexType>
void test_sycl_computations(const Eigen::SyclDevice &sycl_device) {

  IndexType sizeDim1 = 100;
  IndexType sizeDim2 = 10;
  IndexType sizeDim3 = 20;
  array<IndexType, 3> tensorRange = {{sizeDim1, sizeDim2, sizeDim3}};
  Tensor<DataType, 3,DataLayout, IndexType> in1(tensorRange);
  Tensor<DataType, 3,DataLayout, IndexType> in2(tensorRange);
  Tensor<DataType, 3,DataLayout, IndexType> in3(tensorRange);
  Tensor<DataType, 3,DataLayout, IndexType> out(tensorRange);

  in2 = in2.random();
  in3 = in3.random();

  DataType * gpu_in1_data  = static_cast<DataType*>(sycl_device.allocate(in1.size()*sizeof(DataType)));
  DataType * gpu_in2_data  = static_cast<DataType*>(sycl_device.allocate(in2.size()*sizeof(DataType)));
  DataType * gpu_in3_data  = static_cast<DataType*>(sycl_device.allocate(in3.size()*sizeof(DataType)));
  DataType * gpu_out_data =  static_cast<DataType*>(sycl_device.allocate(out.size()*sizeof(DataType)));

  TensorMap<Tensor<DataType, 3, DataLayout, IndexType>> gpu_in1(gpu_in1_data, tensorRange);
  TensorMap<Tensor<DataType, 3, DataLayout, IndexType>> gpu_in2(gpu_in2_data, tensorRange);
  TensorMap<Tensor<DataType, 3, DataLayout, IndexType>> gpu_in3(gpu_in3_data, tensorRange);
  TensorMap<Tensor<DataType, 3, DataLayout, IndexType>> gpu_out(gpu_out_data, tensorRange);

  /// a=1.2f
  gpu_in1.device(sycl_device) = gpu_in1.constant(1.2f);
  sycl_device.memcpyDeviceToHost(in1.data(), gpu_in1_data ,(in1.size())*sizeof(DataType));
  sycl_device.synchronize();

  for (IndexType i = 0; i < sizeDim1; ++i) {
    for (IndexType j = 0; j < sizeDim2; ++j) {
      for (IndexType k = 0; k < sizeDim3; ++k) {
        VERIFY_IS_APPROX(in1(i,j,k), 1.2f);
      }
    }
  }
  printf("a=1.2f Test passed\n");

  /// a=b*1.2f
  gpu_out.device(sycl_device) = gpu_in1 * 1.2f;
  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data ,(out.size())*sizeof(DataType));
  sycl_device.synchronize();

  for (IndexType i = 0; i < sizeDim1; ++i) {
    for (IndexType j = 0; j < sizeDim2; ++j) {
      for (IndexType k = 0; k < sizeDim3; ++k) {
        VERIFY_IS_APPROX(out(i,j,k),
                         in1(i,j,k) * 1.2f);
      }
    }
  }
  printf("a=b*1.2f Test Passed\n");

  /// c=a*b
  sycl_device.memcpyHostToDevice(gpu_in2_data, in2.data(),(in2.size())*sizeof(DataType));
  gpu_out.device(sycl_device) = gpu_in1 * gpu_in2;
  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.size())*sizeof(DataType));
  sycl_device.synchronize();

  for (IndexType i = 0; i < sizeDim1; ++i) {
    for (IndexType j = 0; j < sizeDim2; ++j) {
      for (IndexType k = 0; k < sizeDim3; ++k) {
        VERIFY_IS_APPROX(out(i,j,k),
                         in1(i,j,k) *
                             in2(i,j,k));
      }
    }
  }
  printf("c=a*b Test Passed\n");

  /// c=a+b
  gpu_out.device(sycl_device) = gpu_in1 + gpu_in2;
  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.size())*sizeof(DataType));
  sycl_device.synchronize();
  for (IndexType i = 0; i < sizeDim1; ++i) {
    for (IndexType j = 0; j < sizeDim2; ++j) {
      for (IndexType k = 0; k < sizeDim3; ++k) {
        VERIFY_IS_APPROX(out(i,j,k),
                         in1(i,j,k) +
                             in2(i,j,k));
      }
    }
  }
  printf("c=a+b Test Passed\n");

  /// c=a*a
  gpu_out.device(sycl_device) = gpu_in1 * gpu_in1;
  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.size())*sizeof(DataType));
  sycl_device.synchronize();
  for (IndexType i = 0; i < sizeDim1; ++i) {
    for (IndexType j = 0; j < sizeDim2; ++j) {
      for (IndexType k = 0; k < sizeDim3; ++k) {
        VERIFY_IS_APPROX(out(i,j,k),
                         in1(i,j,k) *
                             in1(i,j,k));
      }
    }
  }
  printf("c= a*a Test Passed\n");

  //a*3.14f + b*2.7f
  gpu_out.device(sycl_device) =  gpu_in1 * gpu_in1.constant(3.14f) + gpu_in2 * gpu_in2.constant(2.7f);
  sycl_device.memcpyDeviceToHost(out.data(),gpu_out_data,(out.size())*sizeof(DataType));
  sycl_device.synchronize();
  for (IndexType i = 0; i < sizeDim1; ++i) {
    for (IndexType j = 0; j < sizeDim2; ++j) {
      for (IndexType k = 0; k < sizeDim3; ++k) {
        VERIFY_IS_APPROX(out(i,j,k),
                         in1(i,j,k) * 3.14f
                       + in2(i,j,k) * 2.7f);
      }
    }
  }
  printf("a*3.14f + b*2.7f Test Passed\n");

  ///d= (a>0.5? b:c)
  sycl_device.memcpyHostToDevice(gpu_in3_data, in3.data(),(in3.size())*sizeof(DataType));
  gpu_out.device(sycl_device) =(gpu_in1 > gpu_in1.constant(0.5f)).select(gpu_in2, gpu_in3);
  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.size())*sizeof(DataType));
  sycl_device.synchronize();
  for (IndexType i = 0; i < sizeDim1; ++i) {
    for (IndexType j = 0; j < sizeDim2; ++j) {
      for (IndexType k = 0; k < sizeDim3; ++k) {
        VERIFY_IS_APPROX(out(i, j, k), (in1(i, j, k) > 0.5f)
                                                ? in2(i, j, k)
                                                : in3(i, j, k));
      }
    }
  }
  printf("d= (a>0.5? b:c) Test Passed\n");
  sycl_device.deallocate(gpu_in1_data);
  sycl_device.deallocate(gpu_in2_data);
  sycl_device.deallocate(gpu_in3_data);
  sycl_device.deallocate(gpu_out_data);
}
template<typename Scalar1, typename Scalar2,  int DataLayout, typename IndexType>
static void test_sycl_cast(const Eigen::SyclDevice& sycl_device){
    IndexType size = 20;
    array<IndexType, 1> tensorRange = {{size}};
    Tensor<Scalar1, 1, DataLayout, IndexType> in(tensorRange);
    Tensor<Scalar2, 1, DataLayout, IndexType> out(tensorRange);
    Tensor<Scalar2, 1, DataLayout, IndexType> out_host(tensorRange);

    in = in.random();

    Scalar1* gpu_in_data  = static_cast<Scalar1*>(sycl_device.allocate(in.size()*sizeof(Scalar1)));
    Scalar2 * gpu_out_data =  static_cast<Scalar2*>(sycl_device.allocate(out.size()*sizeof(Scalar2)));

    TensorMap<Tensor<Scalar1, 1, DataLayout, IndexType>> gpu_in(gpu_in_data, tensorRange);
    TensorMap<Tensor<Scalar2, 1, DataLayout, IndexType>> gpu_out(gpu_out_data, tensorRange);
    sycl_device.memcpyHostToDevice(gpu_in_data, in.data(),(in.size())*sizeof(Scalar1));
    gpu_out.device(sycl_device) = gpu_in. template cast<Scalar2>();
    sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data, out.size()*sizeof(Scalar2));
    out_host = in. template cast<Scalar2>();
    for(IndexType i=0; i< size; i++)
    {
      VERIFY_IS_APPROX(out(i), out_host(i));
    }
    printf("cast Test Passed\n");
    sycl_device.deallocate(gpu_in_data);
    sycl_device.deallocate(gpu_out_data);
}
template<typename DataType, typename dev_Selector> void sycl_computing_test_per_device(dev_Selector s){
  QueueInterface queueInterface(s);
  auto sycl_device = Eigen::SyclDevice(&queueInterface);
  test_sycl_mem_transfers<DataType, RowMajor, int64_t>(sycl_device);
  test_sycl_computations<DataType, RowMajor, int64_t>(sycl_device);
  test_sycl_mem_sync<DataType, RowMajor, int64_t>(sycl_device);
  test_sycl_mem_sync_offsets<DataType, RowMajor, int64_t>(sycl_device);
  test_sycl_memset_offsets<DataType, RowMajor, int64_t>(sycl_device);
  test_sycl_mem_transfers<DataType, ColMajor, int64_t>(sycl_device);
  test_sycl_computations<DataType, ColMajor, int64_t>(sycl_device);
  test_sycl_mem_sync<DataType, ColMajor, int64_t>(sycl_device);
  test_sycl_cast<DataType, int, RowMajor, int64_t>(sycl_device);
  test_sycl_cast<DataType, int, ColMajor, int64_t>(sycl_device);
}

EIGEN_DECLARE_TEST(cxx11_tensor_sycl) {
  for (const auto& device :Eigen::get_sycl_supported_devices()) {
    CALL_SUBTEST(sycl_computing_test_per_device<float>(device));
  }
}
