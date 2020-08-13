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
#define EIGEN_HAS_CONSTEXPR 1

#include "main.h"

#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::array;
using Eigen::SyclDevice;
using Eigen::Tensor;
using Eigen::TensorMap;

template <typename DataType, int Layout, typename DenseIndex>
static void test_sycl_simple_argmax(const Eigen::SyclDevice& sycl_device) {
  Tensor<DataType, 3, Layout, DenseIndex> in(Eigen::array<DenseIndex, 3>{{2, 2, 2}});
  Tensor<DenseIndex, 0, Layout, DenseIndex> out_max;
  Tensor<DenseIndex, 0, Layout, DenseIndex> out_min;
  in.setRandom();
  in *= in.constant(100.0);
  in(0, 0, 0) = -1000.0;
  in(1, 1, 1) = 1000.0;

  std::size_t in_bytes = in.size() * sizeof(DataType);
  std::size_t out_bytes = out_max.size() * sizeof(DenseIndex);

  DataType* d_in = static_cast<DataType*>(sycl_device.allocate(in_bytes));
  DenseIndex* d_out_max = static_cast<DenseIndex*>(sycl_device.allocate(out_bytes));
  DenseIndex* d_out_min = static_cast<DenseIndex*>(sycl_device.allocate(out_bytes));

  Eigen::TensorMap<Eigen::Tensor<DataType, 3, Layout, DenseIndex> > gpu_in(d_in,
                                                                           Eigen::array<DenseIndex, 3>{{2, 2, 2}});
  Eigen::TensorMap<Eigen::Tensor<DenseIndex, 0, Layout, DenseIndex> > gpu_out_max(d_out_max);
  Eigen::TensorMap<Eigen::Tensor<DenseIndex, 0, Layout, DenseIndex> > gpu_out_min(d_out_min);
  sycl_device.memcpyHostToDevice(d_in, in.data(), in_bytes);

  gpu_out_max.device(sycl_device) = gpu_in.argmax();
  gpu_out_min.device(sycl_device) = gpu_in.argmin();

  sycl_device.memcpyDeviceToHost(out_max.data(), d_out_max, out_bytes);
  sycl_device.memcpyDeviceToHost(out_min.data(), d_out_min, out_bytes);

  VERIFY_IS_EQUAL(out_max(), 2 * 2 * 2 - 1);
  VERIFY_IS_EQUAL(out_min(), 0);

  sycl_device.deallocate(d_in);
  sycl_device.deallocate(d_out_max);
  sycl_device.deallocate(d_out_min);
}

template <typename DataType, int DataLayout, typename DenseIndex>
static void test_sycl_argmax_dim(const Eigen::SyclDevice& sycl_device) {
  DenseIndex sizeDim0 = 9;
  DenseIndex sizeDim1 = 3;
  DenseIndex sizeDim2 = 5;
  DenseIndex sizeDim3 = 7;
  Tensor<DataType, 4, DataLayout, DenseIndex> tensor(sizeDim0, sizeDim1, sizeDim2, sizeDim3);

  std::vector<DenseIndex> dims;
  dims.push_back(sizeDim0);
  dims.push_back(sizeDim1);
  dims.push_back(sizeDim2);
  dims.push_back(sizeDim3);
  for (DenseIndex dim = 0; dim < 4; ++dim) {
    array<DenseIndex, 3> out_shape;
    for (DenseIndex d = 0; d < 3; ++d) out_shape[d] = (d < dim) ? dims[d] : dims[d + 1];

    Tensor<DenseIndex, 3, DataLayout, DenseIndex> tensor_arg(out_shape);

    array<DenseIndex, 4> ix;
    for (DenseIndex i = 0; i < sizeDim0; ++i) {
      for (DenseIndex j = 0; j < sizeDim1; ++j) {
        for (DenseIndex k = 0; k < sizeDim2; ++k) {
          for (DenseIndex l = 0; l < sizeDim3; ++l) {
            ix[0] = i;
            ix[1] = j;
            ix[2] = k;
            ix[3] = l;
            // suppose dim == 1, then for all i, k, l, set tensor(i, 0, k, l)
            // = 10.0
            tensor(ix) = (ix[dim] != 0) ? -1.0 : 10.0;
          }
        }
      }
    }

    std::size_t in_bytes = tensor.size() * sizeof(DataType);
    std::size_t out_bytes = tensor_arg.size() * sizeof(DenseIndex);

    DataType* d_in = static_cast<DataType*>(sycl_device.allocate(in_bytes));
    DenseIndex* d_out = static_cast<DenseIndex*>(sycl_device.allocate(out_bytes));

    Eigen::TensorMap<Eigen::Tensor<DataType, 4, DataLayout, DenseIndex> > gpu_in(
        d_in, Eigen::array<DenseIndex, 4>{{sizeDim0, sizeDim1, sizeDim2, sizeDim3}});
    Eigen::TensorMap<Eigen::Tensor<DenseIndex, 3, DataLayout, DenseIndex> > gpu_out(d_out, out_shape);

    sycl_device.memcpyHostToDevice(d_in, tensor.data(), in_bytes);
    gpu_out.device(sycl_device) = gpu_in.argmax(dim);
    sycl_device.memcpyDeviceToHost(tensor_arg.data(), d_out, out_bytes);

    VERIFY_IS_EQUAL(static_cast<size_t>(tensor_arg.size()),
                    size_t(sizeDim0 * sizeDim1 * sizeDim2 * sizeDim3 / tensor.dimension(dim)));

    for (DenseIndex n = 0; n < tensor_arg.size(); ++n) {
      // Expect max to be in the first index of the reduced dimension
      VERIFY_IS_EQUAL(tensor_arg.data()[n], 0);
    }

    sycl_device.synchronize();

    for (DenseIndex i = 0; i < sizeDim0; ++i) {
      for (DenseIndex j = 0; j < sizeDim1; ++j) {
        for (DenseIndex k = 0; k < sizeDim2; ++k) {
          for (DenseIndex l = 0; l < sizeDim3; ++l) {
            ix[0] = i;
            ix[1] = j;
            ix[2] = k;
            ix[3] = l;
            // suppose dim == 1, then for all i, k, l, set tensor(i, 2, k, l) = 20.0
            tensor(ix) = (ix[dim] != tensor.dimension(dim) - 1) ? -1.0 : 20.0;
          }
        }
      }
    }

    sycl_device.memcpyHostToDevice(d_in, tensor.data(), in_bytes);
    gpu_out.device(sycl_device) = gpu_in.argmax(dim);
    sycl_device.memcpyDeviceToHost(tensor_arg.data(), d_out, out_bytes);

    for (DenseIndex n = 0; n < tensor_arg.size(); ++n) {
      // Expect max to be in the last index of the reduced dimension
      VERIFY_IS_EQUAL(tensor_arg.data()[n], tensor.dimension(dim) - 1);
    }
    sycl_device.deallocate(d_in);
    sycl_device.deallocate(d_out);
  }
}

template <typename DataType, int DataLayout, typename DenseIndex>
static void test_sycl_argmin_dim(const Eigen::SyclDevice& sycl_device) {
  DenseIndex sizeDim0 = 9;
  DenseIndex sizeDim1 = 3;
  DenseIndex sizeDim2 = 5;
  DenseIndex sizeDim3 = 7;
  Tensor<DataType, 4, DataLayout, DenseIndex> tensor(sizeDim0, sizeDim1, sizeDim2, sizeDim3);

  std::vector<DenseIndex> dims;
  dims.push_back(sizeDim0);
  dims.push_back(sizeDim1);
  dims.push_back(sizeDim2);
  dims.push_back(sizeDim3);
  for (DenseIndex dim = 0; dim < 4; ++dim) {
    array<DenseIndex, 3> out_shape;
    for (DenseIndex d = 0; d < 3; ++d) out_shape[d] = (d < dim) ? dims[d] : dims[d + 1];

    Tensor<DenseIndex, 3, DataLayout, DenseIndex> tensor_arg(out_shape);

    array<DenseIndex, 4> ix;
    for (DenseIndex i = 0; i < sizeDim0; ++i) {
      for (DenseIndex j = 0; j < sizeDim1; ++j) {
        for (DenseIndex k = 0; k < sizeDim2; ++k) {
          for (DenseIndex l = 0; l < sizeDim3; ++l) {
            ix[0] = i;
            ix[1] = j;
            ix[2] = k;
            ix[3] = l;
            // suppose dim == 1, then for all i, k, l, set tensor(i, 0, k, l) = -10.0
            tensor(ix) = (ix[dim] != 0) ? 1.0 : -10.0;
          }
        }
      }
    }

    std::size_t in_bytes = tensor.size() * sizeof(DataType);
    std::size_t out_bytes = tensor_arg.size() * sizeof(DenseIndex);

    DataType* d_in = static_cast<DataType*>(sycl_device.allocate(in_bytes));
    DenseIndex* d_out = static_cast<DenseIndex*>(sycl_device.allocate(out_bytes));

    Eigen::TensorMap<Eigen::Tensor<DataType, 4, DataLayout, DenseIndex> > gpu_in(
        d_in, Eigen::array<DenseIndex, 4>{{sizeDim0, sizeDim1, sizeDim2, sizeDim3}});
    Eigen::TensorMap<Eigen::Tensor<DenseIndex, 3, DataLayout, DenseIndex> > gpu_out(d_out, out_shape);

    sycl_device.memcpyHostToDevice(d_in, tensor.data(), in_bytes);
    gpu_out.device(sycl_device) = gpu_in.argmin(dim);
    sycl_device.memcpyDeviceToHost(tensor_arg.data(), d_out, out_bytes);

    VERIFY_IS_EQUAL(static_cast<size_t>(tensor_arg.size()),
                    size_t(sizeDim0 * sizeDim1 * sizeDim2 * sizeDim3 / tensor.dimension(dim)));

    for (DenseIndex n = 0; n < tensor_arg.size(); ++n) {
      // Expect max to be in the first index of the reduced dimension
      VERIFY_IS_EQUAL(tensor_arg.data()[n], 0);
    }

    sycl_device.synchronize();

    for (DenseIndex i = 0; i < sizeDim0; ++i) {
      for (DenseIndex j = 0; j < sizeDim1; ++j) {
        for (DenseIndex k = 0; k < sizeDim2; ++k) {
          for (DenseIndex l = 0; l < sizeDim3; ++l) {
            ix[0] = i;
            ix[1] = j;
            ix[2] = k;
            ix[3] = l;
            // suppose dim == 1, then for all i, k, l, set tensor(i, 2, k, l) = -20.0
            tensor(ix) = (ix[dim] != tensor.dimension(dim) - 1) ? 1.0 : -20.0;
          }
        }
      }
    }

    sycl_device.memcpyHostToDevice(d_in, tensor.data(), in_bytes);
    gpu_out.device(sycl_device) = gpu_in.argmin(dim);
    sycl_device.memcpyDeviceToHost(tensor_arg.data(), d_out, out_bytes);

    for (DenseIndex n = 0; n < tensor_arg.size(); ++n) {
      // Expect max to be in the last index of the reduced dimension
      VERIFY_IS_EQUAL(tensor_arg.data()[n], tensor.dimension(dim) - 1);
    }
    sycl_device.deallocate(d_in);
    sycl_device.deallocate(d_out);
  }
}

template <typename DataType, typename Device_Selector>
void sycl_argmax_test_per_device(const Device_Selector& d) {
  QueueInterface queueInterface(d);
  auto sycl_device = Eigen::SyclDevice(&queueInterface);
  test_sycl_simple_argmax<DataType, RowMajor, int64_t>(sycl_device);
  test_sycl_simple_argmax<DataType, ColMajor, int64_t>(sycl_device);
  test_sycl_argmax_dim<DataType, ColMajor, int64_t>(sycl_device);
  test_sycl_argmax_dim<DataType, RowMajor, int64_t>(sycl_device);
  test_sycl_argmin_dim<DataType, ColMajor, int64_t>(sycl_device);
  test_sycl_argmin_dim<DataType, RowMajor, int64_t>(sycl_device);
}

EIGEN_DECLARE_TEST(cxx11_tensor_argmax_sycl) {
  for (const auto& device : Eigen::get_sycl_supported_devices()) {
    CALL_SUBTEST(sycl_argmax_test_per_device<float>(device));
  }
}
