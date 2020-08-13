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
typedef Tensor<float, 1>::DimensionPair DimPair;

template <typename DataType, int DataLayout, typename IndexType>
void test_sycl_cumsum(const Eigen::SyclDevice& sycl_device, IndexType m_size,
                      IndexType k_size, IndexType n_size, int consume_dim,
                      bool exclusive) {
  static const DataType error_threshold = 1e-4f;
  std::cout << "Testing for (" << m_size << "," << k_size << "," << n_size
            << " consume_dim : " << consume_dim << ")" << std::endl;
  Tensor<DataType, 3, DataLayout, IndexType> t_input(m_size, k_size, n_size);
  Tensor<DataType, 3, DataLayout, IndexType> t_result(m_size, k_size, n_size);
  Tensor<DataType, 3, DataLayout, IndexType> t_result_gpu(m_size, k_size,
                                                          n_size);

  t_input.setRandom();
  std::size_t t_input_bytes = t_input.size() * sizeof(DataType);
  std::size_t t_result_bytes = t_result.size() * sizeof(DataType);

  DataType* gpu_data_in =
      static_cast<DataType*>(sycl_device.allocate(t_input_bytes));
  DataType* gpu_data_out =
      static_cast<DataType*>(sycl_device.allocate(t_result_bytes));

  array<IndexType, 3> tensorRange = {{m_size, k_size, n_size}};
  TensorMap<Tensor<DataType, 3, DataLayout, IndexType>> gpu_t_input(
      gpu_data_in, tensorRange);
  TensorMap<Tensor<DataType, 3, DataLayout, IndexType>> gpu_t_result(
      gpu_data_out, tensorRange);
  sycl_device.memcpyHostToDevice(gpu_data_in, t_input.data(), t_input_bytes);
  sycl_device.memcpyHostToDevice(gpu_data_out, t_input.data(), t_input_bytes);

  gpu_t_result.device(sycl_device) = gpu_t_input.cumsum(consume_dim, exclusive);

  t_result = t_input.cumsum(consume_dim, exclusive);

  sycl_device.memcpyDeviceToHost(t_result_gpu.data(), gpu_data_out,
                                 t_result_bytes);
  sycl_device.synchronize();

  for (IndexType i = 0; i < t_result.size(); i++) {
    if (static_cast<DataType>(std::fabs(static_cast<DataType>(
            t_result(i) - t_result_gpu(i)))) < error_threshold) {
      continue;
    }
    if (Eigen::internal::isApprox(t_result(i), t_result_gpu(i),
                                  error_threshold)) {
      continue;
    }
    std::cout << "mismatch detected at index " << i << " CPU : " << t_result(i)
              << " vs SYCL : " << t_result_gpu(i) << std::endl;
    assert(false);
  }
  sycl_device.deallocate(gpu_data_in);
  sycl_device.deallocate(gpu_data_out);
}

template <typename DataType, typename Dev>
void sycl_scan_test_exclusive_dim0_per_device(const Dev& sycl_device) {
  test_sycl_cumsum<DataType, ColMajor, int64_t>(sycl_device, 2049, 1023, 127, 0,
                                                true);
  test_sycl_cumsum<DataType, RowMajor, int64_t>(sycl_device, 2049, 1023, 127, 0,
                                                true);
}
template <typename DataType, typename Dev>
void sycl_scan_test_exclusive_dim1_per_device(const Dev& sycl_device) {
  test_sycl_cumsum<DataType, ColMajor, int64_t>(sycl_device, 1023, 2049, 127, 1,
                                                true);
  test_sycl_cumsum<DataType, RowMajor, int64_t>(sycl_device, 1023, 2049, 127, 1,
                                                true);
}
template <typename DataType, typename Dev>
void sycl_scan_test_exclusive_dim2_per_device(const Dev& sycl_device) {
  test_sycl_cumsum<DataType, ColMajor, int64_t>(sycl_device, 1023, 127, 2049, 2,
                                                true);
  test_sycl_cumsum<DataType, RowMajor, int64_t>(sycl_device, 1023, 127, 2049, 2,
                                                true);
}
template <typename DataType, typename Dev>
void sycl_scan_test_inclusive_dim0_per_device(const Dev& sycl_device) {
  test_sycl_cumsum<DataType, ColMajor, int64_t>(sycl_device, 2049, 1023, 127, 0,
                                                false);
  test_sycl_cumsum<DataType, RowMajor, int64_t>(sycl_device, 2049, 1023, 127, 0,
                                                false);
}
template <typename DataType, typename Dev>
void sycl_scan_test_inclusive_dim1_per_device(const Dev& sycl_device) {
  test_sycl_cumsum<DataType, ColMajor, int64_t>(sycl_device, 1023, 2049, 127, 1,
                                                false);
  test_sycl_cumsum<DataType, RowMajor, int64_t>(sycl_device, 1023, 2049, 127, 1,
                                                false);
}
template <typename DataType, typename Dev>
void sycl_scan_test_inclusive_dim2_per_device(const Dev& sycl_device) {
  test_sycl_cumsum<DataType, ColMajor, int64_t>(sycl_device, 1023, 127, 2049, 2,
                                                false);
  test_sycl_cumsum<DataType, RowMajor, int64_t>(sycl_device, 1023, 127, 2049, 2,
                                                false);
}
EIGEN_DECLARE_TEST(cxx11_tensor_scan_sycl) {
  for (const auto& device : Eigen::get_sycl_supported_devices()) {
    std::cout << "Running on "
              << device.template get_info<cl::sycl::info::device::name>()
              << std::endl;
    QueueInterface queueInterface(device);
    auto sycl_device = Eigen::SyclDevice(&queueInterface);
    CALL_SUBTEST_1(
        sycl_scan_test_exclusive_dim0_per_device<float>(sycl_device));
    CALL_SUBTEST_2(
        sycl_scan_test_exclusive_dim1_per_device<float>(sycl_device));
    CALL_SUBTEST_3(
        sycl_scan_test_exclusive_dim2_per_device<float>(sycl_device));
    CALL_SUBTEST_4(
        sycl_scan_test_inclusive_dim0_per_device<float>(sycl_device));
    CALL_SUBTEST_5(
        sycl_scan_test_inclusive_dim1_per_device<float>(sycl_device));
    CALL_SUBTEST_6(
        sycl_scan_test_inclusive_dim2_per_device<float>(sycl_device));
  }
}
