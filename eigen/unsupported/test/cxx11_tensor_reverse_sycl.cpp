// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015
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
static void test_simple_reverse(const Eigen::SyclDevice& sycl_device) {
  IndexType dim1 = 2;
  IndexType dim2 = 3;
  IndexType dim3 = 5;
  IndexType dim4 = 7;

  array<IndexType, 4> tensorRange = {{dim1, dim2, dim3, dim4}};
  Tensor<DataType, 4, DataLayout, IndexType> tensor(tensorRange);
  Tensor<DataType, 4, DataLayout, IndexType> reversed_tensor(tensorRange);
  tensor.setRandom();

  array<bool, 4> dim_rev;
  dim_rev[0] = false;
  dim_rev[1] = true;
  dim_rev[2] = true;
  dim_rev[3] = false;

  DataType* gpu_in_data = static_cast<DataType*>(
      sycl_device.allocate(tensor.dimensions().TotalSize() * sizeof(DataType)));
  DataType* gpu_out_data = static_cast<DataType*>(sycl_device.allocate(
      reversed_tensor.dimensions().TotalSize() * sizeof(DataType)));

  TensorMap<Tensor<DataType, 4, DataLayout, IndexType> > in_gpu(gpu_in_data,
                                                                tensorRange);
  TensorMap<Tensor<DataType, 4, DataLayout, IndexType> > out_gpu(gpu_out_data,
                                                                 tensorRange);

  sycl_device.memcpyHostToDevice(
      gpu_in_data, tensor.data(),
      (tensor.dimensions().TotalSize()) * sizeof(DataType));
  out_gpu.device(sycl_device) = in_gpu.reverse(dim_rev);
  sycl_device.memcpyDeviceToHost(
      reversed_tensor.data(), gpu_out_data,
      reversed_tensor.dimensions().TotalSize() * sizeof(DataType));
  // Check that the CPU and GPU reductions return the same result.
  for (IndexType i = 0; i < 2; ++i) {
    for (IndexType j = 0; j < 3; ++j) {
      for (IndexType k = 0; k < 5; ++k) {
        for (IndexType l = 0; l < 7; ++l) {
          VERIFY_IS_EQUAL(tensor(i, j, k, l),
                          reversed_tensor(i, 2 - j, 4 - k, l));
        }
      }
    }
  }
  dim_rev[0] = true;
  dim_rev[1] = false;
  dim_rev[2] = false;
  dim_rev[3] = false;

  out_gpu.device(sycl_device) = in_gpu.reverse(dim_rev);
  sycl_device.memcpyDeviceToHost(
      reversed_tensor.data(), gpu_out_data,
      reversed_tensor.dimensions().TotalSize() * sizeof(DataType));

  for (IndexType i = 0; i < 2; ++i) {
    for (IndexType j = 0; j < 3; ++j) {
      for (IndexType k = 0; k < 5; ++k) {
        for (IndexType l = 0; l < 7; ++l) {
          VERIFY_IS_EQUAL(tensor(i, j, k, l), reversed_tensor(1 - i, j, k, l));
        }
      }
    }
  }

  dim_rev[0] = true;
  dim_rev[1] = false;
  dim_rev[2] = false;
  dim_rev[3] = true;
  out_gpu.device(sycl_device) = in_gpu.reverse(dim_rev);
  sycl_device.memcpyDeviceToHost(
      reversed_tensor.data(), gpu_out_data,
      reversed_tensor.dimensions().TotalSize() * sizeof(DataType));

  for (IndexType i = 0; i < 2; ++i) {
    for (IndexType j = 0; j < 3; ++j) {
      for (IndexType k = 0; k < 5; ++k) {
        for (IndexType l = 0; l < 7; ++l) {
          VERIFY_IS_EQUAL(tensor(i, j, k, l),
                          reversed_tensor(1 - i, j, k, 6 - l));
        }
      }
    }
  }

  sycl_device.deallocate(gpu_in_data);
  sycl_device.deallocate(gpu_out_data);
}

template <typename DataType, int DataLayout, typename IndexType>
static void test_expr_reverse(const Eigen::SyclDevice& sycl_device,
                              bool LValue) {
  IndexType dim1 = 2;
  IndexType dim2 = 3;
  IndexType dim3 = 5;
  IndexType dim4 = 7;

  array<IndexType, 4> tensorRange = {{dim1, dim2, dim3, dim4}};
  Tensor<DataType, 4, DataLayout, IndexType> tensor(tensorRange);
  Tensor<DataType, 4, DataLayout, IndexType> expected(tensorRange);
  Tensor<DataType, 4, DataLayout, IndexType> result(tensorRange);
  tensor.setRandom();

  array<bool, 4> dim_rev;
  dim_rev[0] = false;
  dim_rev[1] = true;
  dim_rev[2] = false;
  dim_rev[3] = true;

  DataType* gpu_in_data = static_cast<DataType*>(
      sycl_device.allocate(tensor.dimensions().TotalSize() * sizeof(DataType)));
  DataType* gpu_out_data_expected = static_cast<DataType*>(sycl_device.allocate(
      expected.dimensions().TotalSize() * sizeof(DataType)));
  DataType* gpu_out_data_result = static_cast<DataType*>(
      sycl_device.allocate(result.dimensions().TotalSize() * sizeof(DataType)));

  TensorMap<Tensor<DataType, 4, DataLayout, IndexType> > in_gpu(gpu_in_data,
                                                                tensorRange);
  TensorMap<Tensor<DataType, 4, DataLayout, IndexType> > out_gpu_expected(
      gpu_out_data_expected, tensorRange);
  TensorMap<Tensor<DataType, 4, DataLayout, IndexType> > out_gpu_result(
      gpu_out_data_result, tensorRange);

  sycl_device.memcpyHostToDevice(
      gpu_in_data, tensor.data(),
      (tensor.dimensions().TotalSize()) * sizeof(DataType));

  if (LValue) {
    out_gpu_expected.reverse(dim_rev).device(sycl_device) = in_gpu;
  } else {
    out_gpu_expected.device(sycl_device) = in_gpu.reverse(dim_rev);
  }
  sycl_device.memcpyDeviceToHost(
      expected.data(), gpu_out_data_expected,
      expected.dimensions().TotalSize() * sizeof(DataType));

  array<IndexType, 4> src_slice_dim;
  src_slice_dim[0] = 2;
  src_slice_dim[1] = 3;
  src_slice_dim[2] = 1;
  src_slice_dim[3] = 7;
  array<IndexType, 4> src_slice_start;
  src_slice_start[0] = 0;
  src_slice_start[1] = 0;
  src_slice_start[2] = 0;
  src_slice_start[3] = 0;
  array<IndexType, 4> dst_slice_dim = src_slice_dim;
  array<IndexType, 4> dst_slice_start = src_slice_start;

  for (IndexType i = 0; i < 5; ++i) {
    if (LValue) {
      out_gpu_result.slice(dst_slice_start, dst_slice_dim)
          .reverse(dim_rev)
          .device(sycl_device) = in_gpu.slice(src_slice_start, src_slice_dim);
    } else {
      out_gpu_result.slice(dst_slice_start, dst_slice_dim).device(sycl_device) =
          in_gpu.slice(src_slice_start, src_slice_dim).reverse(dim_rev);
    }
    src_slice_start[2] += 1;
    dst_slice_start[2] += 1;
  }
  sycl_device.memcpyDeviceToHost(
      result.data(), gpu_out_data_result,
      result.dimensions().TotalSize() * sizeof(DataType));

  for (IndexType i = 0; i < expected.dimension(0); ++i) {
    for (IndexType j = 0; j < expected.dimension(1); ++j) {
      for (IndexType k = 0; k < expected.dimension(2); ++k) {
        for (IndexType l = 0; l < expected.dimension(3); ++l) {
          VERIFY_IS_EQUAL(result(i, j, k, l), expected(i, j, k, l));
        }
      }
    }
  }

  dst_slice_start[2] = 0;
  result.setRandom();
  sycl_device.memcpyHostToDevice(
      gpu_out_data_result, result.data(),
      (result.dimensions().TotalSize()) * sizeof(DataType));
  for (IndexType i = 0; i < 5; ++i) {
    if (LValue) {
      out_gpu_result.slice(dst_slice_start, dst_slice_dim)
          .reverse(dim_rev)
          .device(sycl_device) = in_gpu.slice(dst_slice_start, dst_slice_dim);
    } else {
      out_gpu_result.slice(dst_slice_start, dst_slice_dim).device(sycl_device) =
          in_gpu.reverse(dim_rev).slice(dst_slice_start, dst_slice_dim);
    }
    dst_slice_start[2] += 1;
  }
  sycl_device.memcpyDeviceToHost(
      result.data(), gpu_out_data_result,
      result.dimensions().TotalSize() * sizeof(DataType));

  for (IndexType i = 0; i < expected.dimension(0); ++i) {
    for (IndexType j = 0; j < expected.dimension(1); ++j) {
      for (IndexType k = 0; k < expected.dimension(2); ++k) {
        for (IndexType l = 0; l < expected.dimension(3); ++l) {
          VERIFY_IS_EQUAL(result(i, j, k, l), expected(i, j, k, l));
        }
      }
    }
  }
}

template <typename DataType>
void sycl_reverse_test_per_device(const cl::sycl::device& d) {
  QueueInterface queueInterface(d);
  auto sycl_device = Eigen::SyclDevice(&queueInterface);
  test_simple_reverse<DataType, RowMajor, int64_t>(sycl_device);
  test_simple_reverse<DataType, ColMajor, int64_t>(sycl_device);
  test_expr_reverse<DataType, RowMajor, int64_t>(sycl_device, false);
  test_expr_reverse<DataType, ColMajor, int64_t>(sycl_device, false);
  test_expr_reverse<DataType, RowMajor, int64_t>(sycl_device, true);
  test_expr_reverse<DataType, ColMajor, int64_t>(sycl_device, true);
}
EIGEN_DECLARE_TEST(cxx11_tensor_reverse_sycl) {
  for (const auto& device : Eigen::get_sycl_supported_devices()) {
    std::cout << "Running on "
              << device.get_info<cl::sycl::info::device::name>() << std::endl;
    CALL_SUBTEST_1(sycl_reverse_test_per_device<short>(device));
    CALL_SUBTEST_2(sycl_reverse_test_per_device<int>(device));
    CALL_SUBTEST_3(sycl_reverse_test_per_device<unsigned int>(device));
#ifdef EIGEN_SYCL_DOUBLE_SUPPORT
    CALL_SUBTEST_4(sycl_reverse_test_per_device<double>(device));
#endif
    CALL_SUBTEST_5(sycl_reverse_test_per_device<float>(device));
  }
}
