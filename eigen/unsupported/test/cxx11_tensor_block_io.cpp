// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// clang-format off
#include "main.h"
#include <Eigen/CXX11/Tensor>
// clang-format on

// -------------------------------------------------------------------------- //
// A set of tests for TensorBlockIO: copying data between tensor blocks.

template <int NumDims>
static DSizes<Index, NumDims> RandomDims(Index min, Index max) {
  DSizes<Index, NumDims> dims;
  for (int i = 0; i < NumDims; ++i) {
    dims[i] = internal::random<Index>(min, max);
  }
  return DSizes<Index, NumDims>(dims);
}

static internal::TensorBlockShapeType RandomBlockShape() {
  return internal::random<bool>()
         ? internal::TensorBlockShapeType::kUniformAllDims
         : internal::TensorBlockShapeType::kSkewedInnerDims;
}

template <int NumDims>
static size_t RandomTargetBlockSize(const DSizes<Index, NumDims>& dims) {
  return internal::random<size_t>(1, dims.TotalSize());
}

template <int Layout, int NumDims>
static Index GetInputIndex(Index output_index,
                           const array<Index, NumDims>& output_to_input_dim_map,
                           const array<Index, NumDims>& input_strides,
                           const array<Index, NumDims>& output_strides) {
  int input_index = 0;
  if (Layout == ColMajor) {
    for (int i = NumDims - 1; i > 0; --i) {
      const Index idx = output_index / output_strides[i];
      input_index += idx * input_strides[output_to_input_dim_map[i]];
      output_index -= idx * output_strides[i];
    }
    return input_index +
           output_index * input_strides[output_to_input_dim_map[0]];
  } else {
    for (int i = 0; i < NumDims - 1; ++i) {
      const Index idx = output_index / output_strides[i];
      input_index += idx * input_strides[output_to_input_dim_map[i]];
      output_index -= idx * output_strides[i];
    }
    return input_index +
           output_index * input_strides[output_to_input_dim_map[NumDims - 1]];
  }
}

template <typename T, int NumDims, int Layout>
static void test_block_io_copy_data_from_source_to_target() {
  using TensorBlockIO = internal::TensorBlockIO<T, Index, NumDims, Layout>;
  using IODst = typename TensorBlockIO::Dst;
  using IOSrc = typename TensorBlockIO::Src;

  // Generate a random input Tensor.
  DSizes<Index, NumDims> dims = RandomDims<NumDims>(1, 30);
  Tensor<T, NumDims, Layout> input(dims);
  input.setRandom();

  // Write data to an output Tensor.
  Tensor<T, NumDims, Layout> output(dims);

  // Construct a tensor block mapper.
  using TensorBlockMapper =
      internal::TensorBlockMapper<NumDims, Layout, Index>;
  TensorBlockMapper block_mapper(
      dims, {RandomBlockShape(), RandomTargetBlockSize(dims), {0, 0, 0}});

  // We will copy data from input to output through this buffer.
  Tensor<T, NumDims, Layout> block(block_mapper.blockDimensions());

  // Precompute strides for TensorBlockIO::Copy.
  auto input_strides = internal::strides<Layout>(dims);
  auto output_strides = internal::strides<Layout>(dims);

  const T* input_data = input.data();
  T* output_data = output.data();
  T* block_data = block.data();

  for (int i = 0; i < block_mapper.blockCount(); ++i) {
    auto desc = block_mapper.blockDescriptor(i);

    auto blk_dims = desc.dimensions();
    auto blk_strides = internal::strides<Layout>(blk_dims);

    {
      // Read from input into a block buffer.
      IODst dst(blk_dims, blk_strides, block_data, 0);
      IOSrc src(input_strides, input_data, desc.offset());

      TensorBlockIO::Copy(dst, src);
    }

    {
      // Write from block buffer to output.
      IODst dst(blk_dims, output_strides, output_data, desc.offset());
      IOSrc src(blk_strides, block_data, 0);

      TensorBlockIO::Copy(dst, src);
    }
  }

  for (int i = 0; i < dims.TotalSize(); ++i) {
    VERIFY_IS_EQUAL(input_data[i], output_data[i]);
  }
}

template <typename T, int NumDims, int Layout>
static void test_block_io_copy_using_reordered_dimensions() {
  // Generate a random input Tensor.
  DSizes<Index, NumDims> dims = RandomDims<NumDims>(1, 30);
  Tensor<T, NumDims, Layout> input(dims);
  input.setRandom();

  // Create a random dimension re-ordering/shuffle.
  std::vector<int> shuffle;

  for (int i = 0; i < NumDims; ++i) shuffle.push_back(i);
  std::shuffle(shuffle.begin(), shuffle.end(), std::mt19937(g_seed));

  DSizes<Index, NumDims> output_tensor_dims;
  DSizes<Index, NumDims> input_to_output_dim_map;
  DSizes<Index, NumDims> output_to_input_dim_map;
  for (Index i = 0; i < NumDims; ++i) {
    output_tensor_dims[shuffle[i]] = dims[i];
    input_to_output_dim_map[i] = shuffle[i];
    output_to_input_dim_map[shuffle[i]] = i;
  }

  // Write data to an output Tensor.
  Tensor<T, NumDims, Layout> output(output_tensor_dims);

  // Construct a tensor block mapper.
  // NOTE: Tensor block mapper works with shuffled dimensions.
  using TensorBlockMapper =
      internal::TensorBlockMapper<NumDims, Layout, Index>;
  TensorBlockMapper block_mapper(output_tensor_dims,
                                 {RandomBlockShape(),
                                  RandomTargetBlockSize(output_tensor_dims),
                                  {0, 0, 0}});

  // We will copy data from input to output through this buffer.
  Tensor<T, NumDims, Layout> block(block_mapper.blockDimensions());

  // Precompute strides for TensorBlockIO::Copy.
  auto input_strides = internal::strides<Layout>(dims);
  auto output_strides = internal::strides<Layout>(output_tensor_dims);

  const T* input_data = input.data();
  T* output_data = output.data();
  T* block_data = block.data();

  for (Index i = 0; i < block_mapper.blockCount(); ++i) {
    auto desc = block_mapper.blockDescriptor(i);

    const Index first_coeff_index = GetInputIndex<Layout, NumDims>(
        desc.offset(), output_to_input_dim_map, input_strides,
        output_strides);

    // NOTE: Block dimensions are in the same order as output dimensions.

    using TensorBlockIO = internal::TensorBlockIO<T, Index, NumDims, Layout>;
    using IODst = typename TensorBlockIO::Dst;
    using IOSrc = typename TensorBlockIO::Src;

    auto blk_dims = desc.dimensions();
    auto blk_strides = internal::strides<Layout>(blk_dims);

    {
      // Read from input into a block buffer.
      IODst dst(blk_dims, blk_strides, block_data, 0);
      IOSrc src(input_strides, input_data, first_coeff_index);

      // TODO(ezhulenev): Remove when fully switched to TensorBlock.
      DSizes<int, NumDims> dim_map;
      for (int j = 0; j < NumDims; ++j)
        dim_map[j] = static_cast<int>(output_to_input_dim_map[j]);
      TensorBlockIO::Copy(dst, src, /*dst_to_src_dim_map=*/dim_map);
    }

    {
      // We need to convert block dimensions from output to input order.
      auto dst_dims = blk_dims;
      for (int out_dim = 0; out_dim < NumDims; ++out_dim) {
        dst_dims[output_to_input_dim_map[out_dim]] = blk_dims[out_dim];
      }

      // Write from block buffer to output.
      IODst dst(dst_dims, input_strides, output_data, first_coeff_index);
      IOSrc src(blk_strides, block_data, 0);

      // TODO(ezhulenev): Remove when fully switched to TensorBlock.
      DSizes<int, NumDims> dim_map;
      for (int j = 0; j < NumDims; ++j)
        dim_map[j] = static_cast<int>(input_to_output_dim_map[j]);
      TensorBlockIO::Copy(dst, src, /*dst_to_src_dim_map=*/dim_map);
    }
  }

  for (Index i = 0; i < dims.TotalSize(); ++i) {
    VERIFY_IS_EQUAL(input_data[i], output_data[i]);
  }
}

// This is the special case for reading data with reordering, when dimensions
// before/after reordering are the same. Squeezing reads along inner dimensions
// in this case is illegal, because we reorder innermost dimension.
template <int Layout>
static void test_block_io_copy_using_reordered_dimensions_do_not_squeeze() {
  DSizes<Index, 3> tensor_dims(7, 9, 7);
  DSizes<Index, 3> block_dims = tensor_dims;

  DSizes<int, 3> block_to_tensor_dim;
  block_to_tensor_dim[0] = 2;
  block_to_tensor_dim[1] = 1;
  block_to_tensor_dim[2] = 0;

  auto tensor_strides = internal::strides<Layout>(tensor_dims);
  auto block_strides = internal::strides<Layout>(block_dims);

  Tensor<float, 3, Layout> block(block_dims);
  Tensor<float, 3, Layout> tensor(tensor_dims);
  tensor.setRandom();

  float* tensor_data = tensor.data();
  float* block_data = block.data();

  using TensorBlockIO = internal::TensorBlockIO<float, Index, 3, Layout>;
  using IODst = typename TensorBlockIO::Dst;
  using IOSrc = typename TensorBlockIO::Src;

  // Read from a tensor into a block.
  IODst dst(block_dims, block_strides, block_data, 0);
  IOSrc src(tensor_strides, tensor_data, 0);

  TensorBlockIO::Copy(dst, src, /*dst_to_src_dim_map=*/block_to_tensor_dim);

  TensorMap<Tensor<float, 3, Layout> > block_tensor(block_data, block_dims);
  TensorMap<Tensor<float, 3, Layout> > tensor_tensor(tensor_data, tensor_dims);

  for (Index d0 = 0; d0 < tensor_dims[0]; ++d0) {
    for (Index d1 = 0; d1 < tensor_dims[1]; ++d1) {
      for (Index d2 = 0; d2 < tensor_dims[2]; ++d2) {
        float block_value = block_tensor(d2, d1, d0);
        float tensor_value = tensor_tensor(d0, d1, d2);
        VERIFY_IS_EQUAL(block_value, tensor_value);
      }
    }
  }
}

// This is the special case for reading data with reordering, when dimensions
// before/after reordering are the same. Squeezing reads in this case is allowed
// because we reorder outer dimensions.
template <int Layout>
static void test_block_io_copy_using_reordered_dimensions_squeeze() {
  DSizes<Index, 4> tensor_dims(7, 5, 9, 9);
  DSizes<Index, 4> block_dims = tensor_dims;

  DSizes<int, 4> block_to_tensor_dim;
  block_to_tensor_dim[0] = 0;
  block_to_tensor_dim[1] = 1;
  block_to_tensor_dim[2] = 3;
  block_to_tensor_dim[3] = 2;

  auto tensor_strides = internal::strides<Layout>(tensor_dims);
  auto block_strides = internal::strides<Layout>(block_dims);

  Tensor<float, 4, Layout> block(block_dims);
  Tensor<float, 4, Layout> tensor(tensor_dims);
  tensor.setRandom();

  float* tensor_data = tensor.data();
  float* block_data = block.data();

  using TensorBlockIO = internal::TensorBlockIO<float, Index, 4, Layout>;
  using IODst = typename TensorBlockIO::Dst;
  using IOSrc = typename TensorBlockIO::Src;

  // Read from a tensor into a block.
  IODst dst(block_dims, block_strides, block_data, 0);
  IOSrc src(tensor_strides, tensor_data, 0);

  TensorBlockIO::Copy(dst, src, /*dst_to_src_dim_map=*/block_to_tensor_dim);

  TensorMap<Tensor<float, 4, Layout> > block_tensor(block_data, block_dims);
  TensorMap<Tensor<float, 4, Layout> > tensor_tensor(tensor_data, tensor_dims);

  for (Index d0 = 0; d0 < tensor_dims[0]; ++d0) {
    for (Index d1 = 0; d1 < tensor_dims[1]; ++d1) {
      for (Index d2 = 0; d2 < tensor_dims[2]; ++d2) {
        for (Index d3 = 0; d3 < tensor_dims[3]; ++d3) {
          float block_value = block_tensor(d0, d1, d3, d2);
          float tensor_value = tensor_tensor(d0, d1, d2, d3);
          VERIFY_IS_EQUAL(block_value, tensor_value);
        }
      }
    }
  }
}

template <int Layout>
static void test_block_io_zero_stride() {
  DSizes<Index, 5> rnd_dims = RandomDims<5>(1, 30);

  DSizes<Index, 5> input_tensor_dims = rnd_dims;
  input_tensor_dims[0] = 1;
  input_tensor_dims[2] = 1;
  input_tensor_dims[4] = 1;

  Tensor<float, 5, Layout> input(input_tensor_dims);
  input.setRandom();

  DSizes<Index, 5> output_tensor_dims = rnd_dims;

  auto input_tensor_strides = internal::strides<Layout>(input_tensor_dims);
  auto output_tensor_strides = internal::strides<Layout>(output_tensor_dims);

  auto input_tensor_strides_with_zeros = input_tensor_strides;
  input_tensor_strides_with_zeros[0] = 0;
  input_tensor_strides_with_zeros[2] = 0;
  input_tensor_strides_with_zeros[4] = 0;

  Tensor<float, 5, Layout> output(output_tensor_dims);
  output.setRandom();

  using TensorBlockIO = internal::TensorBlockIO<float, Index, 5, Layout>;
  using IODst = typename TensorBlockIO::Dst;
  using IOSrc = typename TensorBlockIO::Src;

  // Write data from input to output with broadcasting in dims [0, 2, 4].
  IODst dst(output_tensor_dims, output_tensor_strides, output.data(), 0);
  IOSrc src(input_tensor_strides_with_zeros, input.data(), 0);
  TensorBlockIO::Copy(dst, src);

  for (int i = 0; i < output_tensor_dims[0]; ++i) {
    for (int j = 0; j < output_tensor_dims[1]; ++j) {
      for (int k = 0; k < output_tensor_dims[2]; ++k) {
        for (int l = 0; l < output_tensor_dims[3]; ++l) {
          for (int m = 0; m < output_tensor_dims[4]; ++m) {
            float input_value = input(0, j, 0, l, 0);
            float output_value = output(i, j, k, l, m);
            VERIFY_IS_EQUAL(input_value, output_value);
          }
        }
      }
    }
  }
}

template <int Layout>
static void test_block_io_squeeze_ones() {
  using TensorBlockIO = internal::TensorBlockIO<float, Index, 5, Layout>;
  using IODst = typename TensorBlockIO::Dst;
  using IOSrc = typename TensorBlockIO::Src;

  // Total size > 1.
  {
    DSizes<Index, 5> block_sizes(1, 2, 1, 2, 1);
    auto strides = internal::strides<Layout>(block_sizes);

    // Create a random input tensor.
    Tensor<float, 5> input(block_sizes);
    input.setRandom();

    Tensor<float, 5> output(block_sizes);

    IODst dst(block_sizes, strides, output.data(), 0);
    IOSrc src(strides, input.data());
    TensorBlockIO::Copy(dst, src);

    for (Index i = 0; i < block_sizes.TotalSize(); ++i) {
      VERIFY_IS_EQUAL(output.data()[i], input.data()[i]);
    }
  }

  // Total size == 1.
  {
    DSizes<Index, 5> block_sizes(1, 1, 1, 1, 1);
    auto strides = internal::strides<Layout>(block_sizes);

    // Create a random input tensor.
    Tensor<float, 5> input(block_sizes);
    input.setRandom();

    Tensor<float, 5> output(block_sizes);

    IODst dst(block_sizes, strides, output.data(), 0);
    IOSrc src(strides, input.data());
    TensorBlockIO::Copy(dst, src);

    for (Index i = 0; i < block_sizes.TotalSize(); ++i) {
      VERIFY_IS_EQUAL(output.data()[i], input.data()[i]);
    }
  }
}

#define CALL_SUBTESTS(NAME)                   \
  CALL_SUBTEST((NAME<float, 1, RowMajor>())); \
  CALL_SUBTEST((NAME<float, 2, RowMajor>())); \
  CALL_SUBTEST((NAME<float, 4, RowMajor>())); \
  CALL_SUBTEST((NAME<float, 5, RowMajor>())); \
  CALL_SUBTEST((NAME<float, 1, ColMajor>())); \
  CALL_SUBTEST((NAME<float, 2, ColMajor>())); \
  CALL_SUBTEST((NAME<float, 4, ColMajor>())); \
  CALL_SUBTEST((NAME<float, 5, ColMajor>())); \
  CALL_SUBTEST((NAME<bool, 1, RowMajor>())); \
  CALL_SUBTEST((NAME<bool, 2, RowMajor>())); \
  CALL_SUBTEST((NAME<bool, 4, RowMajor>())); \
  CALL_SUBTEST((NAME<bool, 5, RowMajor>())); \
  CALL_SUBTEST((NAME<bool, 1, ColMajor>())); \
  CALL_SUBTEST((NAME<bool, 2, ColMajor>())); \
  CALL_SUBTEST((NAME<bool, 4, ColMajor>())); \
  CALL_SUBTEST((NAME<bool, 5, ColMajor>()))

EIGEN_DECLARE_TEST(cxx11_tensor_block_io) {
  // clang-format off
  CALL_SUBTESTS(test_block_io_copy_data_from_source_to_target);
  CALL_SUBTESTS(test_block_io_copy_using_reordered_dimensions);

  CALL_SUBTEST(test_block_io_copy_using_reordered_dimensions_do_not_squeeze<RowMajor>());
  CALL_SUBTEST(test_block_io_copy_using_reordered_dimensions_do_not_squeeze<ColMajor>());

  CALL_SUBTEST(test_block_io_copy_using_reordered_dimensions_squeeze<RowMajor>());
  CALL_SUBTEST(test_block_io_copy_using_reordered_dimensions_squeeze<ColMajor>());

  CALL_SUBTEST(test_block_io_zero_stride<RowMajor>());
  CALL_SUBTEST(test_block_io_zero_stride<ColMajor>());

  CALL_SUBTEST(test_block_io_squeeze_ones<RowMajor>());
  CALL_SUBTEST(test_block_io_squeeze_ones<ColMajor>());
  // clang-format on
}
