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
#define EIGEN_HAS_CONSTEXPR 1

#include "main.h"

#include <unsupported/Eigen/CXX11/Tensor>

template <typename DataType, int DataLayout, typename IndexType>
static void test_full_reductions_sum_sycl(
    const Eigen::SyclDevice& sycl_device) {
  const IndexType num_rows = 753;
  const IndexType num_cols = 537;
  array<IndexType, 2> tensorRange = {{num_rows, num_cols}};

  array<IndexType, 2> outRange = {{1, 1}};

  Tensor<DataType, 2, DataLayout, IndexType> in(tensorRange);
  Tensor<DataType, 2, DataLayout, IndexType> full_redux(outRange);
  Tensor<DataType, 2, DataLayout, IndexType> full_redux_gpu(outRange);

  in.setRandom();
  auto dim = DSizes<IndexType, 2>(1, 1);
  full_redux = in.sum().reshape(dim);

  DataType* gpu_in_data = static_cast<DataType*>(
      sycl_device.allocate(in.dimensions().TotalSize() * sizeof(DataType)));
  DataType* gpu_out_data = (DataType*)sycl_device.allocate(
      sizeof(DataType) * (full_redux_gpu.dimensions().TotalSize()));

  TensorMap<Tensor<DataType, 2, DataLayout, IndexType>> in_gpu(gpu_in_data,
                                                               tensorRange);
  TensorMap<Tensor<DataType, 2, DataLayout, IndexType>> out_gpu(gpu_out_data,
                                                                outRange);
  sycl_device.memcpyHostToDevice(
      gpu_in_data, in.data(), (in.dimensions().TotalSize()) * sizeof(DataType));
  out_gpu.device(sycl_device) = in_gpu.sum().reshape(dim);
  sycl_device.memcpyDeviceToHost(
      full_redux_gpu.data(), gpu_out_data,
      (full_redux_gpu.dimensions().TotalSize()) * sizeof(DataType));
  // Check that the CPU and GPU reductions return the same result.
  std::cout << "SYCL FULL :" << full_redux_gpu(0, 0)
            << ", CPU FULL: " << full_redux(0, 0) << "\n";
  VERIFY_IS_APPROX(full_redux_gpu(0, 0), full_redux(0, 0));
  sycl_device.deallocate(gpu_in_data);
  sycl_device.deallocate(gpu_out_data);
}

template <typename DataType, int DataLayout, typename IndexType>
static void test_full_reductions_sum_with_offset_sycl(
    const Eigen::SyclDevice& sycl_device) {
  using data_tensor = Tensor<DataType, 2, DataLayout, IndexType>;
  using scalar_tensor = Tensor<DataType, 0, DataLayout, IndexType>;
  const IndexType num_rows = 64;
  const IndexType num_cols = 64;
  array<IndexType, 2> tensor_range = {{num_rows, num_cols}};
  const IndexType n_elems = internal::array_prod(tensor_range);

  data_tensor in(tensor_range);
  scalar_tensor full_redux;
  scalar_tensor full_redux_gpu;

  in.setRandom();
  array<IndexType, 2> tensor_offset_range(tensor_range);
  tensor_offset_range[0] -= 1;

  const IndexType offset = 64;
  TensorMap<data_tensor> in_offset(in.data() + offset, tensor_offset_range);
  full_redux = in_offset.sum();

  DataType* gpu_in_data =
      static_cast<DataType*>(sycl_device.allocate(n_elems * sizeof(DataType)));
  DataType* gpu_out_data =
      static_cast<DataType*>(sycl_device.allocate(sizeof(DataType)));

  TensorMap<data_tensor> in_gpu(gpu_in_data + offset, tensor_offset_range);
  TensorMap<scalar_tensor> out_gpu(gpu_out_data);
  sycl_device.memcpyHostToDevice(gpu_in_data, in.data(),
                                 n_elems * sizeof(DataType));
  out_gpu.device(sycl_device) = in_gpu.sum();
  sycl_device.memcpyDeviceToHost(full_redux_gpu.data(), gpu_out_data,
                                 sizeof(DataType));

  // Check that the CPU and GPU reductions return the same result.
  VERIFY_IS_APPROX(full_redux_gpu(), full_redux());

  sycl_device.deallocate(gpu_in_data);
  sycl_device.deallocate(gpu_out_data);
}

template <typename DataType, int DataLayout, typename IndexType>
static void test_full_reductions_max_sycl(
    const Eigen::SyclDevice& sycl_device) {
  const IndexType num_rows = 4096;
  const IndexType num_cols = 4096;
  array<IndexType, 2> tensorRange = {{num_rows, num_cols}};

  Tensor<DataType, 2, DataLayout, IndexType> in(tensorRange);
  Tensor<DataType, 0, DataLayout, IndexType> full_redux;
  Tensor<DataType, 0, DataLayout, IndexType> full_redux_gpu;

  in.setRandom();

  full_redux = in.maximum();

  DataType* gpu_in_data = static_cast<DataType*>(
      sycl_device.allocate(in.dimensions().TotalSize() * sizeof(DataType)));
  DataType* gpu_out_data = (DataType*)sycl_device.allocate(sizeof(DataType));

  TensorMap<Tensor<DataType, 2, DataLayout, IndexType>> in_gpu(gpu_in_data,
                                                               tensorRange);
  TensorMap<Tensor<DataType, 0, DataLayout, IndexType>> out_gpu(gpu_out_data);
  sycl_device.memcpyHostToDevice(
      gpu_in_data, in.data(), (in.dimensions().TotalSize()) * sizeof(DataType));
  out_gpu.device(sycl_device) = in_gpu.maximum();
  sycl_device.memcpyDeviceToHost(full_redux_gpu.data(), gpu_out_data,
                                 sizeof(DataType));
  VERIFY_IS_APPROX(full_redux_gpu(), full_redux());
  sycl_device.deallocate(gpu_in_data);
  sycl_device.deallocate(gpu_out_data);
}

template <typename DataType, int DataLayout, typename IndexType>
static void test_full_reductions_max_with_offset_sycl(
    const Eigen::SyclDevice& sycl_device) {
  using data_tensor = Tensor<DataType, 2, DataLayout, IndexType>;
  using scalar_tensor = Tensor<DataType, 0, DataLayout, IndexType>;
  const IndexType num_rows = 64;
  const IndexType num_cols = 64;
  array<IndexType, 2> tensor_range = {{num_rows, num_cols}};
  const IndexType n_elems = internal::array_prod(tensor_range);

  data_tensor in(tensor_range);
  scalar_tensor full_redux;
  scalar_tensor full_redux_gpu;

  in.setRandom();
  array<IndexType, 2> tensor_offset_range(tensor_range);
  tensor_offset_range[0] -= 1;
  // Set the initial value to be the max.
  // As we don't include this in the reduction the result should not be 2.
  in(0) = static_cast<DataType>(2);

  const IndexType offset = 64;
  TensorMap<data_tensor> in_offset(in.data() + offset, tensor_offset_range);
  full_redux = in_offset.maximum();
  VERIFY_IS_NOT_EQUAL(full_redux(), in(0));

  DataType* gpu_in_data =
      static_cast<DataType*>(sycl_device.allocate(n_elems * sizeof(DataType)));
  DataType* gpu_out_data =
      static_cast<DataType*>(sycl_device.allocate(sizeof(DataType)));

  TensorMap<data_tensor> in_gpu(gpu_in_data + offset, tensor_offset_range);
  TensorMap<scalar_tensor> out_gpu(gpu_out_data);
  sycl_device.memcpyHostToDevice(gpu_in_data, in.data(),
                                 n_elems * sizeof(DataType));
  out_gpu.device(sycl_device) = in_gpu.maximum();
  sycl_device.memcpyDeviceToHost(full_redux_gpu.data(), gpu_out_data,
                                 sizeof(DataType));

  // Check that the CPU and GPU reductions return the same result.
  VERIFY_IS_APPROX(full_redux_gpu(), full_redux());

  sycl_device.deallocate(gpu_in_data);
  sycl_device.deallocate(gpu_out_data);
}

template <typename DataType, int DataLayout, typename IndexType>
static void test_full_reductions_mean_sycl(
    const Eigen::SyclDevice& sycl_device) {
  const IndexType num_rows = 4096;
  const IndexType num_cols = 4096;
  array<IndexType, 2> tensorRange = {{num_rows, num_cols}};
  array<IndexType, 1> argRange = {{num_cols}};
  Eigen::array<IndexType, 1> red_axis;
  red_axis[0] = 0;
  //  red_axis[1]=1;
  Tensor<DataType, 2, DataLayout, IndexType> in(tensorRange);
  Tensor<DataType, 2, DataLayout, IndexType> in_arg1(tensorRange);
  Tensor<DataType, 2, DataLayout, IndexType> in_arg2(tensorRange);
  Tensor<bool, 1, DataLayout, IndexType> out_arg_cpu(argRange);
  Tensor<bool, 1, DataLayout, IndexType> out_arg_gpu(argRange);
  Tensor<bool, 1, DataLayout, IndexType> out_arg_gpu_helper(argRange);
  Tensor<DataType, 0, DataLayout, IndexType> full_redux;
  Tensor<DataType, 0, DataLayout, IndexType> full_redux_gpu;

  in.setRandom();
  in_arg1.setRandom();
  in_arg2.setRandom();

  DataType* gpu_in_data = static_cast<DataType*>(
      sycl_device.allocate(in.dimensions().TotalSize() * sizeof(DataType)));
  DataType* gpu_in_arg1_data = static_cast<DataType*>(sycl_device.allocate(
      in_arg1.dimensions().TotalSize() * sizeof(DataType)));
  DataType* gpu_in_arg2_data = static_cast<DataType*>(sycl_device.allocate(
      in_arg2.dimensions().TotalSize() * sizeof(DataType)));
  bool* gpu_out_arg__gpu_helper_data = static_cast<bool*>(sycl_device.allocate(
      out_arg_gpu.dimensions().TotalSize() * sizeof(DataType)));
  bool* gpu_out_arg_data = static_cast<bool*>(sycl_device.allocate(
      out_arg_gpu.dimensions().TotalSize() * sizeof(DataType)));

  DataType* gpu_out_data = (DataType*)sycl_device.allocate(sizeof(DataType));

  TensorMap<Tensor<DataType, 2, DataLayout, IndexType>> in_gpu(gpu_in_data,
                                                               tensorRange);
  TensorMap<Tensor<DataType, 2, DataLayout, IndexType>> in_Arg1_gpu(
      gpu_in_arg1_data, tensorRange);
  TensorMap<Tensor<DataType, 2, DataLayout, IndexType>> in_Arg2_gpu(
      gpu_in_arg2_data, tensorRange);
  TensorMap<Tensor<bool, 1, DataLayout, IndexType>> out_Argout_gpu(
      gpu_out_arg_data, argRange);
  TensorMap<Tensor<bool, 1, DataLayout, IndexType>> out_Argout_gpu_helper(
      gpu_out_arg__gpu_helper_data, argRange);
  TensorMap<Tensor<DataType, 0, DataLayout, IndexType>> out_gpu(gpu_out_data);

  // CPU VERSION
  out_arg_cpu =
      (in_arg1.argmax(1) == in_arg2.argmax(1))
          .select(out_arg_cpu.constant(true), out_arg_cpu.constant(false));
  full_redux = (out_arg_cpu.template cast<float>())
                   .reduce(red_axis, Eigen::internal::MeanReducer<DataType>());

  // GPU VERSION
  sycl_device.memcpyHostToDevice(
      gpu_in_data, in.data(), (in.dimensions().TotalSize()) * sizeof(DataType));
  sycl_device.memcpyHostToDevice(
      gpu_in_arg1_data, in_arg1.data(),
      (in_arg1.dimensions().TotalSize()) * sizeof(DataType));
  sycl_device.memcpyHostToDevice(
      gpu_in_arg2_data, in_arg2.data(),
      (in_arg2.dimensions().TotalSize()) * sizeof(DataType));
  out_Argout_gpu_helper.device(sycl_device) =
      (in_Arg1_gpu.argmax(1) == in_Arg2_gpu.argmax(1));
  out_Argout_gpu.device(sycl_device) =
      (out_Argout_gpu_helper)
          .select(out_Argout_gpu.constant(true),
                  out_Argout_gpu.constant(false));
  out_gpu.device(sycl_device) =
      (out_Argout_gpu.template cast<float>())
          .reduce(red_axis, Eigen::internal::MeanReducer<DataType>());
  sycl_device.memcpyDeviceToHost(full_redux_gpu.data(), gpu_out_data,
                                 sizeof(DataType));
  // Check that the CPU and GPU reductions return the same result.
  std::cout << "SYCL : " << full_redux_gpu() << " , CPU : " << full_redux()
            << '\n';
  VERIFY_IS_EQUAL(full_redux_gpu(), full_redux());
  sycl_device.deallocate(gpu_in_data);
  sycl_device.deallocate(gpu_in_arg1_data);
  sycl_device.deallocate(gpu_in_arg2_data);
  sycl_device.deallocate(gpu_out_arg__gpu_helper_data);
  sycl_device.deallocate(gpu_out_arg_data);
  sycl_device.deallocate(gpu_out_data);
}

template <typename DataType, int DataLayout, typename IndexType>
static void test_full_reductions_mean_with_offset_sycl(
    const Eigen::SyclDevice& sycl_device) {
  using data_tensor = Tensor<DataType, 2, DataLayout, IndexType>;
  using scalar_tensor = Tensor<DataType, 0, DataLayout, IndexType>;
  const IndexType num_rows = 64;
  const IndexType num_cols = 64;
  array<IndexType, 2> tensor_range = {{num_rows, num_cols}};
  const IndexType n_elems = internal::array_prod(tensor_range);

  data_tensor in(tensor_range);
  scalar_tensor full_redux;
  scalar_tensor full_redux_gpu;

  in.setRandom();
  array<IndexType, 2> tensor_offset_range(tensor_range);
  tensor_offset_range[0] -= 1;

  const IndexType offset = 64;
  TensorMap<data_tensor> in_offset(in.data() + offset, tensor_offset_range);
  full_redux = in_offset.mean();
  VERIFY_IS_NOT_EQUAL(full_redux(), in(0));

  DataType* gpu_in_data =
      static_cast<DataType*>(sycl_device.allocate(n_elems * sizeof(DataType)));
  DataType* gpu_out_data =
      static_cast<DataType*>(sycl_device.allocate(sizeof(DataType)));

  TensorMap<data_tensor> in_gpu(gpu_in_data + offset, tensor_offset_range);
  TensorMap<scalar_tensor> out_gpu(gpu_out_data);
  sycl_device.memcpyHostToDevice(gpu_in_data, in.data(),
                                 n_elems * sizeof(DataType));
  out_gpu.device(sycl_device) = in_gpu.mean();
  sycl_device.memcpyDeviceToHost(full_redux_gpu.data(), gpu_out_data,
                                 sizeof(DataType));

  // Check that the CPU and GPU reductions return the same result.
  VERIFY_IS_APPROX(full_redux_gpu(), full_redux());

  sycl_device.deallocate(gpu_in_data);
  sycl_device.deallocate(gpu_out_data);
}

template <typename DataType, int DataLayout, typename IndexType>
static void test_full_reductions_mean_with_odd_offset_sycl(
    const Eigen::SyclDevice& sycl_device) {
  // This is a particular case which illustrates a possible problem when the
  // number of local threads in a workgroup is even, but is not a power of two.
  using data_tensor = Tensor<DataType, 1, DataLayout, IndexType>;
  using scalar_tensor = Tensor<DataType, 0, DataLayout, IndexType>;
  // 2177 = (17 * 128) + 1 gives rise to 18 local threads.
  // 8708 = 4 * 2177 = 4 * (17 * 128) + 4 uses 18 vectorised local threads.
  const IndexType n_elems = 8707;
  array<IndexType, 1> tensor_range = {{n_elems}};

  data_tensor in(tensor_range);
  DataType full_redux;
  DataType full_redux_gpu;
  TensorMap<scalar_tensor> red_cpu(&full_redux);
  TensorMap<scalar_tensor> red_gpu(&full_redux_gpu);

  const DataType const_val = static_cast<DataType>(0.6391);
  in = in.constant(const_val);

  Eigen::IndexList<Eigen::type2index<0>> red_axis;
  red_cpu = in.reduce(red_axis, Eigen::internal::MeanReducer<DataType>());
  VERIFY_IS_APPROX(const_val, red_cpu());

  DataType* gpu_in_data =
      static_cast<DataType*>(sycl_device.allocate(n_elems * sizeof(DataType)));
  DataType* gpu_out_data =
      static_cast<DataType*>(sycl_device.allocate(sizeof(DataType)));

  TensorMap<data_tensor> in_gpu(gpu_in_data, tensor_range);
  TensorMap<scalar_tensor> out_gpu(gpu_out_data);
  sycl_device.memcpyHostToDevice(gpu_in_data, in.data(),
                                 n_elems * sizeof(DataType));
  out_gpu.device(sycl_device) =
      in_gpu.reduce(red_axis, Eigen::internal::MeanReducer<DataType>());
  sycl_device.memcpyDeviceToHost(red_gpu.data(), gpu_out_data,
                                 sizeof(DataType));

  // Check that the CPU and GPU reductions return the same result.
  VERIFY_IS_APPROX(full_redux_gpu, full_redux);

  sycl_device.deallocate(gpu_in_data);
  sycl_device.deallocate(gpu_out_data);
}

template <typename DataType, int DataLayout, typename IndexType>
static void test_full_reductions_min_sycl(
    const Eigen::SyclDevice& sycl_device) {
  const IndexType num_rows = 876;
  const IndexType num_cols = 953;
  array<IndexType, 2> tensorRange = {{num_rows, num_cols}};

  Tensor<DataType, 2, DataLayout, IndexType> in(tensorRange);
  Tensor<DataType, 0, DataLayout, IndexType> full_redux;
  Tensor<DataType, 0, DataLayout, IndexType> full_redux_gpu;

  in.setRandom();

  full_redux = in.minimum();

  DataType* gpu_in_data = static_cast<DataType*>(
      sycl_device.allocate(in.dimensions().TotalSize() * sizeof(DataType)));
  DataType* gpu_out_data = (DataType*)sycl_device.allocate(sizeof(DataType));

  TensorMap<Tensor<DataType, 2, DataLayout, IndexType>> in_gpu(gpu_in_data,
                                                               tensorRange);
  TensorMap<Tensor<DataType, 0, DataLayout, IndexType>> out_gpu(gpu_out_data);

  sycl_device.memcpyHostToDevice(
      gpu_in_data, in.data(), (in.dimensions().TotalSize()) * sizeof(DataType));
  out_gpu.device(sycl_device) = in_gpu.minimum();
  sycl_device.memcpyDeviceToHost(full_redux_gpu.data(), gpu_out_data,
                                 sizeof(DataType));
  // Check that the CPU and GPU reductions return the same result.
  VERIFY_IS_APPROX(full_redux_gpu(), full_redux());
  sycl_device.deallocate(gpu_in_data);
  sycl_device.deallocate(gpu_out_data);
}

template <typename DataType, int DataLayout, typename IndexType>
static void test_full_reductions_min_with_offset_sycl(
    const Eigen::SyclDevice& sycl_device) {
  using data_tensor = Tensor<DataType, 2, DataLayout, IndexType>;
  using scalar_tensor = Tensor<DataType, 0, DataLayout, IndexType>;
  const IndexType num_rows = 64;
  const IndexType num_cols = 64;
  array<IndexType, 2> tensor_range = {{num_rows, num_cols}};
  const IndexType n_elems = internal::array_prod(tensor_range);

  data_tensor in(tensor_range);
  scalar_tensor full_redux;
  scalar_tensor full_redux_gpu;

  in.setRandom();
  array<IndexType, 2> tensor_offset_range(tensor_range);
  tensor_offset_range[0] -= 1;
  // Set the initial value to be the min.
  // As we don't include this in the reduction the result should not be -2.
  in(0) = static_cast<DataType>(-2);

  const IndexType offset = 64;
  TensorMap<data_tensor> in_offset(in.data() + offset, tensor_offset_range);
  full_redux = in_offset.minimum();
  VERIFY_IS_NOT_EQUAL(full_redux(), in(0));

  DataType* gpu_in_data =
      static_cast<DataType*>(sycl_device.allocate(n_elems * sizeof(DataType)));
  DataType* gpu_out_data =
      static_cast<DataType*>(sycl_device.allocate(sizeof(DataType)));

  TensorMap<data_tensor> in_gpu(gpu_in_data + offset, tensor_offset_range);
  TensorMap<scalar_tensor> out_gpu(gpu_out_data);
  sycl_device.memcpyHostToDevice(gpu_in_data, in.data(),
                                 n_elems * sizeof(DataType));
  out_gpu.device(sycl_device) = in_gpu.minimum();
  sycl_device.memcpyDeviceToHost(full_redux_gpu.data(), gpu_out_data,
                                 sizeof(DataType));

  // Check that the CPU and GPU reductions return the same result.
  VERIFY_IS_APPROX(full_redux_gpu(), full_redux());

  sycl_device.deallocate(gpu_in_data);
  sycl_device.deallocate(gpu_out_data);
}
template <typename DataType, int DataLayout, typename IndexType>
static void test_first_dim_reductions_max_sycl(
    const Eigen::SyclDevice& sycl_device) {
  IndexType dim_x = 145;
  IndexType dim_y = 1;
  IndexType dim_z = 67;

  array<IndexType, 3> tensorRange = {{dim_x, dim_y, dim_z}};
  Eigen::array<IndexType, 1> red_axis;
  red_axis[0] = 0;
  array<IndexType, 2> reduced_tensorRange = {{dim_y, dim_z}};

  Tensor<DataType, 3, DataLayout, IndexType> in(tensorRange);
  Tensor<DataType, 2, DataLayout, IndexType> redux(reduced_tensorRange);
  Tensor<DataType, 2, DataLayout, IndexType> redux_gpu(reduced_tensorRange);

  in.setRandom();

  redux = in.maximum(red_axis);

  DataType* gpu_in_data = static_cast<DataType*>(
      sycl_device.allocate(in.dimensions().TotalSize() * sizeof(DataType)));
  DataType* gpu_out_data = static_cast<DataType*>(sycl_device.allocate(
      redux_gpu.dimensions().TotalSize() * sizeof(DataType)));

  TensorMap<Tensor<DataType, 3, DataLayout, IndexType>> in_gpu(gpu_in_data,
                                                               tensorRange);
  TensorMap<Tensor<DataType, 2, DataLayout, IndexType>> out_gpu(
      gpu_out_data, reduced_tensorRange);

  sycl_device.memcpyHostToDevice(
      gpu_in_data, in.data(), (in.dimensions().TotalSize()) * sizeof(DataType));
  out_gpu.device(sycl_device) = in_gpu.maximum(red_axis);
  sycl_device.memcpyDeviceToHost(
      redux_gpu.data(), gpu_out_data,
      redux_gpu.dimensions().TotalSize() * sizeof(DataType));

  // Check that the CPU and GPU reductions return the same result.
  for (IndexType j = 0; j < reduced_tensorRange[0]; j++)
    for (IndexType k = 0; k < reduced_tensorRange[1]; k++)
      VERIFY_IS_APPROX(redux_gpu(j, k), redux(j, k));

  sycl_device.deallocate(gpu_in_data);
  sycl_device.deallocate(gpu_out_data);
}

template <typename DataType, int DataLayout, typename IndexType>
static void test_first_dim_reductions_max_with_offset_sycl(
    const Eigen::SyclDevice& sycl_device) {
  using data_tensor = Tensor<DataType, 2, DataLayout, IndexType>;
  using reduced_tensor = Tensor<DataType, 1, DataLayout, IndexType>;

  const IndexType num_rows = 64;
  const IndexType num_cols = 64;
  array<IndexType, 2> tensor_range = {{num_rows, num_cols}};
  array<IndexType, 1> reduced_range = {{num_cols}};
  const IndexType n_elems = internal::array_prod(tensor_range);
  const IndexType n_reduced = num_cols;

  data_tensor in(tensor_range);
  reduced_tensor redux;
  reduced_tensor redux_gpu(reduced_range);

  in.setRandom();
  array<IndexType, 2> tensor_offset_range(tensor_range);
  tensor_offset_range[0] -= 1;
  // Set maximum value outside of the considered range.
  for (IndexType i = 0; i < n_reduced; i++) {
    in(i) = static_cast<DataType>(2);
  }

  Eigen::array<IndexType, 1> red_axis;
  red_axis[0] = 0;

  const IndexType offset = 64;
  TensorMap<data_tensor> in_offset(in.data() + offset, tensor_offset_range);
  redux = in_offset.maximum(red_axis);
  for (IndexType i = 0; i < n_reduced; i++) {
    VERIFY_IS_NOT_EQUAL(redux(i), in(i));
  }

  DataType* gpu_in_data =
      static_cast<DataType*>(sycl_device.allocate(n_elems * sizeof(DataType)));
  DataType* gpu_out_data = static_cast<DataType*>(
      sycl_device.allocate(n_reduced * sizeof(DataType)));

  TensorMap<data_tensor> in_gpu(gpu_in_data + offset, tensor_offset_range);
  TensorMap<reduced_tensor> out_gpu(gpu_out_data, reduced_range);
  sycl_device.memcpyHostToDevice(gpu_in_data, in.data(),
                                 n_elems * sizeof(DataType));
  out_gpu.device(sycl_device) = in_gpu.maximum(red_axis);
  sycl_device.memcpyDeviceToHost(redux_gpu.data(), gpu_out_data,
                                 n_reduced * sizeof(DataType));

  // Check that the CPU and GPU reductions return the same result.
  for (IndexType i = 0; i < n_reduced; i++) {
    VERIFY_IS_APPROX(redux_gpu(i), redux(i));
  }

  sycl_device.deallocate(gpu_in_data);
  sycl_device.deallocate(gpu_out_data);
}

template <typename DataType, int DataLayout, typename IndexType>
static void test_last_dim_reductions_max_with_offset_sycl(
    const Eigen::SyclDevice& sycl_device) {
  using data_tensor = Tensor<DataType, 2, DataLayout, IndexType>;
  using reduced_tensor = Tensor<DataType, 1, DataLayout, IndexType>;

  const IndexType num_rows = 64;
  const IndexType num_cols = 64;
  array<IndexType, 2> tensor_range = {{num_rows, num_cols}};
  array<IndexType, 1> full_reduced_range = {{num_rows}};
  array<IndexType, 1> reduced_range = {{num_rows - 1}};
  const IndexType n_elems = internal::array_prod(tensor_range);
  const IndexType n_reduced = reduced_range[0];

  data_tensor in(tensor_range);
  reduced_tensor redux(full_reduced_range);
  reduced_tensor redux_gpu(reduced_range);

  in.setRandom();
  redux.setZero();
  array<IndexType, 2> tensor_offset_range(tensor_range);
  tensor_offset_range[0] -= 1;
  // Set maximum value outside of the considered range.
  for (IndexType i = 0; i < n_reduced; i++) {
    in(i) = static_cast<DataType>(2);
  }

  Eigen::array<IndexType, 1> red_axis;
  red_axis[0] = 1;

  const IndexType offset = 64;
  // Introduce an offset in both the input and the output.
  TensorMap<data_tensor> in_offset(in.data() + offset, tensor_offset_range);
  TensorMap<reduced_tensor> red_offset(redux.data() + 1, reduced_range);
  red_offset = in_offset.maximum(red_axis);

  // Check that the first value hasn't been changed and that the reduced values
  // are not equal to the previously set maximum in the input outside the range.
  VERIFY_IS_EQUAL(redux(0), static_cast<DataType>(0));
  for (IndexType i = 0; i < n_reduced; i++) {
    VERIFY_IS_NOT_EQUAL(red_offset(i), in(i));
  }

  DataType* gpu_in_data =
      static_cast<DataType*>(sycl_device.allocate(n_elems * sizeof(DataType)));
  DataType* gpu_out_data = static_cast<DataType*>(
      sycl_device.allocate((n_reduced + 1) * sizeof(DataType)));

  TensorMap<data_tensor> in_gpu(gpu_in_data + offset, tensor_offset_range);
  TensorMap<reduced_tensor> out_gpu(gpu_out_data + 1, reduced_range);
  sycl_device.memcpyHostToDevice(gpu_in_data, in.data(),
                                 n_elems * sizeof(DataType));
  out_gpu.device(sycl_device) = in_gpu.maximum(red_axis);
  sycl_device.memcpyDeviceToHost(redux_gpu.data(), out_gpu.data(),
                                 n_reduced * sizeof(DataType));

  // Check that the CPU and GPU reductions return the same result.
  for (IndexType i = 0; i < n_reduced; i++) {
    VERIFY_IS_APPROX(redux_gpu(i), red_offset(i));
  }

  sycl_device.deallocate(gpu_in_data);
  sycl_device.deallocate(gpu_out_data);
}

template <typename DataType, int DataLayout, typename IndexType>
static void test_first_dim_reductions_sum_sycl(
    const Eigen::SyclDevice& sycl_device, IndexType dim_x, IndexType dim_y) {
  array<IndexType, 2> tensorRange = {{dim_x, dim_y}};
  Eigen::array<IndexType, 1> red_axis;
  red_axis[0] = 0;
  array<IndexType, 1> reduced_tensorRange = {{dim_y}};

  Tensor<DataType, 2, DataLayout, IndexType> in(tensorRange);
  Tensor<DataType, 1, DataLayout, IndexType> redux(reduced_tensorRange);
  Tensor<DataType, 1, DataLayout, IndexType> redux_gpu(reduced_tensorRange);

  in.setRandom();
  redux = in.sum(red_axis);

  DataType* gpu_in_data = static_cast<DataType*>(
      sycl_device.allocate(in.dimensions().TotalSize() * sizeof(DataType)));
  DataType* gpu_out_data = static_cast<DataType*>(sycl_device.allocate(
      redux_gpu.dimensions().TotalSize() * sizeof(DataType)));

  TensorMap<Tensor<DataType, 2, DataLayout, IndexType>> in_gpu(gpu_in_data,
                                                               tensorRange);
  TensorMap<Tensor<DataType, 1, DataLayout, IndexType>> out_gpu(
      gpu_out_data, reduced_tensorRange);

  sycl_device.memcpyHostToDevice(
      gpu_in_data, in.data(), (in.dimensions().TotalSize()) * sizeof(DataType));
  out_gpu.device(sycl_device) = in_gpu.sum(red_axis);
  sycl_device.memcpyDeviceToHost(
      redux_gpu.data(), gpu_out_data,
      redux_gpu.dimensions().TotalSize() * sizeof(DataType));

  // Check that the CPU and GPU reductions return the same result.
  for (IndexType i = 0; i < redux.size(); i++) {
    VERIFY_IS_APPROX(redux_gpu.data()[i], redux.data()[i]);
  }
  sycl_device.deallocate(gpu_in_data);
  sycl_device.deallocate(gpu_out_data);
}

template <typename DataType, int DataLayout, typename IndexType>
static void test_first_dim_reductions_mean_sycl(
    const Eigen::SyclDevice& sycl_device) {
  IndexType dim_x = 145;
  IndexType dim_y = 1;
  IndexType dim_z = 67;

  array<IndexType, 3> tensorRange = {{dim_x, dim_y, dim_z}};
  Eigen::array<IndexType, 1> red_axis;
  red_axis[0] = 0;
  array<IndexType, 2> reduced_tensorRange = {{dim_y, dim_z}};

  Tensor<DataType, 3, DataLayout, IndexType> in(tensorRange);
  Tensor<DataType, 2, DataLayout, IndexType> redux(reduced_tensorRange);
  Tensor<DataType, 2, DataLayout, IndexType> redux_gpu(reduced_tensorRange);

  in.setRandom();

  redux = in.mean(red_axis);

  DataType* gpu_in_data = static_cast<DataType*>(
      sycl_device.allocate(in.dimensions().TotalSize() * sizeof(DataType)));
  DataType* gpu_out_data = static_cast<DataType*>(sycl_device.allocate(
      redux_gpu.dimensions().TotalSize() * sizeof(DataType)));

  TensorMap<Tensor<DataType, 3, DataLayout, IndexType>> in_gpu(gpu_in_data,
                                                               tensorRange);
  TensorMap<Tensor<DataType, 2, DataLayout, IndexType>> out_gpu(
      gpu_out_data, reduced_tensorRange);

  sycl_device.memcpyHostToDevice(
      gpu_in_data, in.data(), (in.dimensions().TotalSize()) * sizeof(DataType));
  out_gpu.device(sycl_device) = in_gpu.mean(red_axis);
  sycl_device.memcpyDeviceToHost(
      redux_gpu.data(), gpu_out_data,
      redux_gpu.dimensions().TotalSize() * sizeof(DataType));

  // Check that the CPU and GPU reductions return the same result.
  for (IndexType j = 0; j < reduced_tensorRange[0]; j++)
    for (IndexType k = 0; k < reduced_tensorRange[1]; k++)
      VERIFY_IS_APPROX(redux_gpu(j, k), redux(j, k));

  sycl_device.deallocate(gpu_in_data);
  sycl_device.deallocate(gpu_out_data);
}

template <typename DataType, int DataLayout, typename IndexType>
static void test_last_dim_reductions_mean_sycl(
    const Eigen::SyclDevice& sycl_device) {
  IndexType dim_x = 64;
  IndexType dim_y = 1;
  IndexType dim_z = 32;

  array<IndexType, 3> tensorRange = {{dim_x, dim_y, dim_z}};
  Eigen::array<IndexType, 1> red_axis;
  red_axis[0] = 2;
  array<IndexType, 2> reduced_tensorRange = {{dim_x, dim_y}};

  Tensor<DataType, 3, DataLayout, IndexType> in(tensorRange);
  Tensor<DataType, 2, DataLayout, IndexType> redux(reduced_tensorRange);
  Tensor<DataType, 2, DataLayout, IndexType> redux_gpu(reduced_tensorRange);

  in.setRandom();

  redux = in.mean(red_axis);

  DataType* gpu_in_data = static_cast<DataType*>(
      sycl_device.allocate(in.dimensions().TotalSize() * sizeof(DataType)));
  DataType* gpu_out_data = static_cast<DataType*>(sycl_device.allocate(
      redux_gpu.dimensions().TotalSize() * sizeof(DataType)));

  TensorMap<Tensor<DataType, 3, DataLayout, IndexType>> in_gpu(gpu_in_data,
                                                               tensorRange);
  TensorMap<Tensor<DataType, 2, DataLayout, IndexType>> out_gpu(
      gpu_out_data, reduced_tensorRange);

  sycl_device.memcpyHostToDevice(
      gpu_in_data, in.data(), (in.dimensions().TotalSize()) * sizeof(DataType));
  out_gpu.device(sycl_device) = in_gpu.mean(red_axis);
  sycl_device.memcpyDeviceToHost(
      redux_gpu.data(), gpu_out_data,
      redux_gpu.dimensions().TotalSize() * sizeof(DataType));
  // Check that the CPU and GPU reductions return the same result.
  for (IndexType j = 0; j < reduced_tensorRange[0]; j++)
    for (IndexType k = 0; k < reduced_tensorRange[1]; k++)
      VERIFY_IS_APPROX(redux_gpu(j, k), redux(j, k));

  sycl_device.deallocate(gpu_in_data);
  sycl_device.deallocate(gpu_out_data);
}

template <typename DataType, int DataLayout, typename IndexType>
static void test_last_dim_reductions_sum_sycl(
    const Eigen::SyclDevice& sycl_device) {
  IndexType dim_x = 64;
  IndexType dim_y = 1;
  IndexType dim_z = 32;

  array<IndexType, 3> tensorRange = {{dim_x, dim_y, dim_z}};
  Eigen::array<IndexType, 1> red_axis;
  red_axis[0] = 2;
  array<IndexType, 2> reduced_tensorRange = {{dim_x, dim_y}};

  Tensor<DataType, 3, DataLayout, IndexType> in(tensorRange);
  Tensor<DataType, 2, DataLayout, IndexType> redux(reduced_tensorRange);
  Tensor<DataType, 2, DataLayout, IndexType> redux_gpu(reduced_tensorRange);

  in.setRandom();

  redux = in.sum(red_axis);

  DataType* gpu_in_data = static_cast<DataType*>(
      sycl_device.allocate(in.dimensions().TotalSize() * sizeof(DataType)));
  DataType* gpu_out_data = static_cast<DataType*>(sycl_device.allocate(
      redux_gpu.dimensions().TotalSize() * sizeof(DataType)));

  TensorMap<Tensor<DataType, 3, DataLayout, IndexType>> in_gpu(gpu_in_data,
                                                               tensorRange);
  TensorMap<Tensor<DataType, 2, DataLayout, IndexType>> out_gpu(
      gpu_out_data, reduced_tensorRange);

  sycl_device.memcpyHostToDevice(
      gpu_in_data, in.data(), (in.dimensions().TotalSize()) * sizeof(DataType));
  out_gpu.device(sycl_device) = in_gpu.sum(red_axis);
  sycl_device.memcpyDeviceToHost(
      redux_gpu.data(), gpu_out_data,
      redux_gpu.dimensions().TotalSize() * sizeof(DataType));
  // Check that the CPU and GPU reductions return the same result.
  for (IndexType j = 0; j < reduced_tensorRange[0]; j++)
    for (IndexType k = 0; k < reduced_tensorRange[1]; k++)
      VERIFY_IS_APPROX(redux_gpu(j, k), redux(j, k));

  sycl_device.deallocate(gpu_in_data);
  sycl_device.deallocate(gpu_out_data);
}

template <typename DataType, int DataLayout, typename IndexType>
static void test_last_reductions_sum_sycl(
    const Eigen::SyclDevice& sycl_device) {
  auto tensorRange = Sizes<64, 32>(64, 32);
  // auto red_axis =  Sizes<0,1>(0,1);
  Eigen::IndexList<Eigen::type2index<1>> red_axis;
  auto reduced_tensorRange = Sizes<64>(64);
  TensorFixedSize<DataType, Sizes<64, 32>, DataLayout> in_fix;
  TensorFixedSize<DataType, Sizes<64>, DataLayout> redux_fix;
  TensorFixedSize<DataType, Sizes<64>, DataLayout> redux_gpu_fix;

  in_fix.setRandom();

  redux_fix = in_fix.sum(red_axis);

  DataType* gpu_in_data = static_cast<DataType*>(
      sycl_device.allocate(in_fix.dimensions().TotalSize() * sizeof(DataType)));
  DataType* gpu_out_data = static_cast<DataType*>(sycl_device.allocate(
      redux_gpu_fix.dimensions().TotalSize() * sizeof(DataType)));

  TensorMap<TensorFixedSize<DataType, Sizes<64, 32>, DataLayout>> in_gpu_fix(
      gpu_in_data, tensorRange);
  TensorMap<TensorFixedSize<DataType, Sizes<64>, DataLayout>> out_gpu_fix(
      gpu_out_data, reduced_tensorRange);

  sycl_device.memcpyHostToDevice(
      gpu_in_data, in_fix.data(),
      (in_fix.dimensions().TotalSize()) * sizeof(DataType));
  out_gpu_fix.device(sycl_device) = in_gpu_fix.sum(red_axis);
  sycl_device.memcpyDeviceToHost(
      redux_gpu_fix.data(), gpu_out_data,
      redux_gpu_fix.dimensions().TotalSize() * sizeof(DataType));
  // Check that the CPU and GPU reductions return the same result.
  for (IndexType j = 0; j < reduced_tensorRange[0]; j++) {
    VERIFY_IS_APPROX(redux_gpu_fix(j), redux_fix(j));
  }

  sycl_device.deallocate(gpu_in_data);
  sycl_device.deallocate(gpu_out_data);
}

template <typename DataType, int DataLayout, typename IndexType>
static void test_last_reductions_mean_sycl(
    const Eigen::SyclDevice& sycl_device) {
  auto tensorRange = Sizes<64, 32>(64, 32);
  Eigen::IndexList<Eigen::type2index<1>> red_axis;
  auto reduced_tensorRange = Sizes<64>(64);
  TensorFixedSize<DataType, Sizes<64, 32>, DataLayout> in_fix;
  TensorFixedSize<DataType, Sizes<64>, DataLayout> redux_fix;
  TensorFixedSize<DataType, Sizes<64>, DataLayout> redux_gpu_fix;

  in_fix.setRandom();
  redux_fix = in_fix.mean(red_axis);

  DataType* gpu_in_data = static_cast<DataType*>(
      sycl_device.allocate(in_fix.dimensions().TotalSize() * sizeof(DataType)));
  DataType* gpu_out_data = static_cast<DataType*>(sycl_device.allocate(
      redux_gpu_fix.dimensions().TotalSize() * sizeof(DataType)));

  TensorMap<TensorFixedSize<DataType, Sizes<64, 32>, DataLayout>> in_gpu_fix(
      gpu_in_data, tensorRange);
  TensorMap<TensorFixedSize<DataType, Sizes<64>, DataLayout>> out_gpu_fix(
      gpu_out_data, reduced_tensorRange);

  sycl_device.memcpyHostToDevice(
      gpu_in_data, in_fix.data(),
      (in_fix.dimensions().TotalSize()) * sizeof(DataType));
  out_gpu_fix.device(sycl_device) = in_gpu_fix.mean(red_axis);
  sycl_device.memcpyDeviceToHost(
      redux_gpu_fix.data(), gpu_out_data,
      redux_gpu_fix.dimensions().TotalSize() * sizeof(DataType));
  sycl_device.synchronize();
  // Check that the CPU and GPU reductions return the same result.
  for (IndexType j = 0; j < reduced_tensorRange[0]; j++) {
    VERIFY_IS_APPROX(redux_gpu_fix(j), redux_fix(j));
  }

  sycl_device.deallocate(gpu_in_data);
  sycl_device.deallocate(gpu_out_data);
}

// SYCL supports a generic case of reduction where the accumulator is a
// different type than the input data This is an example on how to get if a
// Tensor contains nan and/or inf in one reduction
template <typename InT, typename OutT>
struct CustomReducer {
  static const bool PacketAccess = false;
  static const bool IsStateful = false;

  static constexpr OutT InfBit = 1;
  static constexpr OutT NanBit = 2;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const InT x,
                                                    OutT* accum) const {
    if (Eigen::numext::isinf(x))
      *accum |= InfBit;
    else if (Eigen::numext::isnan(x))
      *accum |= NanBit;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const OutT x,
                                                    OutT* accum) const {
    *accum |= x;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE OutT initialize() const {
    return OutT(0);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE OutT finalize(const OutT accum) const {
    return accum;
  }
};

template <typename DataType, typename AccumType, int DataLayout,
          typename IndexType>
static void test_full_reductions_custom_sycl(
    const Eigen::SyclDevice& sycl_device) {
  constexpr IndexType InSize = 64;
  auto tensorRange = Sizes<InSize>(InSize);
  Eigen::IndexList<Eigen::type2index<0>> dims;
  auto reduced_tensorRange = Sizes<>();
  TensorFixedSize<DataType, Sizes<InSize>, DataLayout> in_fix;
  TensorFixedSize<AccumType, Sizes<>, DataLayout> redux_gpu_fix;

  CustomReducer<DataType, AccumType> reducer;

  in_fix.setRandom();

  size_t in_size_bytes = in_fix.dimensions().TotalSize() * sizeof(DataType);
  DataType* gpu_in_data =
      static_cast<DataType*>(sycl_device.allocate(in_size_bytes));
  AccumType* gpu_out_data =
      static_cast<AccumType*>(sycl_device.allocate(sizeof(AccumType)));

  TensorMap<TensorFixedSize<DataType, Sizes<InSize>, DataLayout>> in_gpu_fix(
      gpu_in_data, tensorRange);
  TensorMap<TensorFixedSize<AccumType, Sizes<>, DataLayout>> out_gpu_fix(
      gpu_out_data, reduced_tensorRange);

  sycl_device.memcpyHostToDevice(gpu_in_data, in_fix.data(), in_size_bytes);
  out_gpu_fix.device(sycl_device) = in_gpu_fix.reduce(dims, reducer);
  sycl_device.memcpyDeviceToHost(redux_gpu_fix.data(), gpu_out_data,
                                 sizeof(AccumType));
  VERIFY_IS_EQUAL(redux_gpu_fix(0), AccumType(0));

  sycl_device.deallocate(gpu_in_data);
  sycl_device.deallocate(gpu_out_data);
}

template <typename DataType, typename Dev>
void sycl_reduction_test_full_per_device(const Dev& sycl_device) {
  test_full_reductions_sum_sycl<DataType, RowMajor, int64_t>(sycl_device);
  test_full_reductions_sum_sycl<DataType, ColMajor, int64_t>(sycl_device);
  test_full_reductions_min_sycl<DataType, ColMajor, int64_t>(sycl_device);
  test_full_reductions_min_sycl<DataType, RowMajor, int64_t>(sycl_device);
  test_full_reductions_max_sycl<DataType, ColMajor, int64_t>(sycl_device);
  test_full_reductions_max_sycl<DataType, RowMajor, int64_t>(sycl_device);

  test_full_reductions_mean_sycl<DataType, ColMajor, int64_t>(sycl_device);
  test_full_reductions_mean_sycl<DataType, RowMajor, int64_t>(sycl_device);
  test_full_reductions_custom_sycl<DataType, int, RowMajor, int64_t>(
      sycl_device);
  test_full_reductions_custom_sycl<DataType, int, ColMajor, int64_t>(
      sycl_device);
  sycl_device.synchronize();
}

template <typename DataType, typename Dev>
void sycl_reduction_full_offset_per_device(const Dev& sycl_device) {
  test_full_reductions_sum_with_offset_sycl<DataType, RowMajor, int64_t>(
      sycl_device);
  test_full_reductions_sum_with_offset_sycl<DataType, ColMajor, int64_t>(
      sycl_device);
  test_full_reductions_min_with_offset_sycl<DataType, RowMajor, int64_t>(
      sycl_device);
  test_full_reductions_min_with_offset_sycl<DataType, ColMajor, int64_t>(
      sycl_device);
  test_full_reductions_max_with_offset_sycl<DataType, ColMajor, int64_t>(
      sycl_device);
  test_full_reductions_max_with_offset_sycl<DataType, RowMajor, int64_t>(
      sycl_device);
  test_full_reductions_mean_with_offset_sycl<DataType, RowMajor, int64_t>(
      sycl_device);
  test_full_reductions_mean_with_offset_sycl<DataType, ColMajor, int64_t>(
      sycl_device);
  test_full_reductions_mean_with_odd_offset_sycl<DataType, RowMajor, int64_t>(
      sycl_device);
  sycl_device.synchronize();
}

template <typename DataType, typename Dev>
void sycl_reduction_test_first_dim_per_device(const Dev& sycl_device) {
  test_first_dim_reductions_sum_sycl<DataType, ColMajor, int64_t>(sycl_device,
                                                                  4197, 4097);
  test_first_dim_reductions_sum_sycl<DataType, RowMajor, int64_t>(sycl_device,
                                                                  4197, 4097);
  test_first_dim_reductions_sum_sycl<DataType, RowMajor, int64_t>(sycl_device,
                                                                  129, 8);
  test_first_dim_reductions_max_sycl<DataType, RowMajor, int64_t>(sycl_device);
  test_first_dim_reductions_max_with_offset_sycl<DataType, RowMajor, int64_t>(
      sycl_device);
  sycl_device.synchronize();
}

template <typename DataType, typename Dev>
void sycl_reduction_test_last_dim_per_device(const Dev& sycl_device) {
  test_last_dim_reductions_sum_sycl<DataType, RowMajor, int64_t>(sycl_device);
  test_last_dim_reductions_max_with_offset_sycl<DataType, RowMajor, int64_t>(
      sycl_device);
  test_last_reductions_sum_sycl<DataType, ColMajor, int64_t>(sycl_device);
  test_last_reductions_sum_sycl<DataType, RowMajor, int64_t>(sycl_device);
  test_last_reductions_mean_sycl<DataType, ColMajor, int64_t>(sycl_device);
  test_last_reductions_mean_sycl<DataType, RowMajor, int64_t>(sycl_device);
  sycl_device.synchronize();
}

EIGEN_DECLARE_TEST(cxx11_tensor_reduction_sycl) {
  for (const auto& device : Eigen::get_sycl_supported_devices()) {
    std::cout << "Running on "
              << device.template get_info<cl::sycl::info::device::name>()
              << std::endl;
    QueueInterface queueInterface(device);
    auto sycl_device = Eigen::SyclDevice(&queueInterface);
    CALL_SUBTEST_1(sycl_reduction_test_full_per_device<float>(sycl_device));
    CALL_SUBTEST_2(sycl_reduction_full_offset_per_device<float>(sycl_device));
    CALL_SUBTEST_3(
        sycl_reduction_test_first_dim_per_device<float>(sycl_device));
    CALL_SUBTEST_4(sycl_reduction_test_last_dim_per_device<float>(sycl_device));
  }
}
