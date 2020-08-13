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

using Eigen::internal::TensorBlockDescriptor;
using Eigen::internal::TensorExecutor;

// -------------------------------------------------------------------------- //
// Utility functions to generate random tensors, blocks, and evaluate them.

template <int NumDims>
static DSizes<Index, NumDims> RandomDims(Index min, Index max) {
  DSizes<Index, NumDims> dims;
  for (int i = 0; i < NumDims; ++i) {
    dims[i] = internal::random<Index>(min, max);
  }
  return DSizes<Index, NumDims>(dims);
}

// Block offsets and extents allows to construct a TensorSlicingOp corresponding
// to a TensorBlockDescriptor.
template <int NumDims>
struct TensorBlockParams {
  DSizes<Index, NumDims> offsets;
  DSizes<Index, NumDims> sizes;
  TensorBlockDescriptor<NumDims, Index> desc;
};

template <int Layout, int NumDims>
static TensorBlockParams<NumDims> RandomBlock(DSizes<Index, NumDims> dims,
                                              Index min, Index max) {
  // Choose random offsets and sizes along all tensor dimensions.
  DSizes<Index, NumDims> offsets(RandomDims<NumDims>(min, max));
  DSizes<Index, NumDims> sizes(RandomDims<NumDims>(min, max));

  // Make sure that offset + size do not overflow dims.
  for (int i = 0; i < NumDims; ++i) {
    offsets[i] = numext::mini(dims[i] - 1, offsets[i]);
    sizes[i] = numext::mini(sizes[i], dims[i] - offsets[i]);
  }

  Index offset = 0;
  DSizes<Index, NumDims> strides = Eigen::internal::strides<Layout>(dims);
  for (int i = 0; i < NumDims; ++i) {
    offset += strides[i] * offsets[i];
  }

  return {offsets, sizes, TensorBlockDescriptor<NumDims, Index>(offset, sizes)};
}

// Generate block with block sizes skewed towards inner dimensions. This type of
// block is required for evaluating broadcast expressions.
template <int Layout, int NumDims>
static TensorBlockParams<NumDims> SkewedInnerBlock(
    DSizes<Index, NumDims> dims) {
  using BlockMapper = internal::TensorBlockMapper<NumDims, Layout, Index>;
  BlockMapper block_mapper(dims,
                           {internal::TensorBlockShapeType::kSkewedInnerDims,
                            internal::random<size_t>(1, dims.TotalSize()),
                            {0, 0, 0}});

  Index total_blocks = block_mapper.blockCount();
  Index block_index = internal::random<Index>(0, total_blocks - 1);
  auto block = block_mapper.blockDescriptor(block_index);
  DSizes<Index, NumDims> sizes = block.dimensions();

  auto strides = internal::strides<Layout>(dims);
  DSizes<Index, NumDims> offsets;

  // Compute offsets for the first block coefficient.
  Index index = block.offset();
  if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
    for (int i = NumDims - 1; i > 0; --i) {
      const Index idx = index / strides[i];
      index -= idx * strides[i];
      offsets[i] = idx;
    }
    if (NumDims > 0) offsets[0] = index;
  } else {
    for (int i = 0; i < NumDims - 1; ++i) {
      const Index idx = index / strides[i];
      index -= idx * strides[i];
      offsets[i] = idx;
    }
    if (NumDims > 0) offsets[NumDims - 1] = index;
  }

  return {offsets, sizes, block};
}

template <int NumDims>
static TensorBlockParams<NumDims> FixedSizeBlock(DSizes<Index, NumDims> dims) {
  DSizes<Index, NumDims> offsets;
  for (int i = 0; i < NumDims; ++i) offsets[i] = 0;

  return {offsets, dims, TensorBlockDescriptor<NumDims, Index>(0, dims)};
}

inline Eigen::IndexList<Index, Eigen::type2index<1>> NByOne(Index n) {
  Eigen::IndexList<Index, Eigen::type2index<1>> ret;
  ret.set(0, n);
  return ret;
}
inline Eigen::IndexList<Eigen::type2index<1>, Index> OneByM(Index m) {
  Eigen::IndexList<Eigen::type2index<1>, Index> ret;
  ret.set(1, m);
  return ret;
}

// -------------------------------------------------------------------------- //
// Verify that block expression evaluation produces the same result as a
// TensorSliceOp (reading a tensor block is same to taking a tensor slice).

template <typename T, int NumDims, int Layout, typename Expression,
          typename GenBlockParams>
static void VerifyBlockEvaluator(Expression expr, GenBlockParams gen_block) {
  using Device = DefaultDevice;
  auto d = Device();

  // Scratch memory allocator for block evaluation.
  typedef internal::TensorBlockScratchAllocator<Device> TensorBlockScratch;
  TensorBlockScratch scratch(d);

  // TensorEvaluator is needed to produce tensor blocks of the expression.
  auto eval = TensorEvaluator<const decltype(expr), Device>(expr, d);
  eval.evalSubExprsIfNeeded(nullptr);

  // Choose a random offsets, sizes and TensorBlockDescriptor.
  TensorBlockParams<NumDims> block_params = gen_block();

  // Evaluate TensorBlock expression into a tensor.
  Tensor<T, NumDims, Layout> block(block_params.desc.dimensions());

  // Dimensions for the potential destination buffer.
  DSizes<Index, NumDims> dst_dims;
  if (internal::random<bool>()) {
    dst_dims = block_params.desc.dimensions();
  } else {
    for (int i = 0; i < NumDims; ++i) {
      Index extent = internal::random<Index>(0, 5);
      dst_dims[i] = block_params.desc.dimension(i) + extent;
    }
  }

  // Maybe use this tensor as a block desc destination.
  Tensor<T, NumDims, Layout> dst(dst_dims);
  dst.setZero();
  if (internal::random<bool>()) {
    block_params.desc.template AddDestinationBuffer<Layout>(
        dst.data(), internal::strides<Layout>(dst.dimensions()));
  }

  const bool root_of_expr = internal::random<bool>();
  auto tensor_block = eval.block(block_params.desc, scratch, root_of_expr);

  if (tensor_block.kind() == internal::TensorBlockKind::kMaterializedInOutput) {
    // Copy data from destination buffer.
    if (dimensions_match(dst.dimensions(), block.dimensions())) {
      block = dst;
    } else {
      DSizes<Index, NumDims> offsets;
      for (int i = 0; i < NumDims; ++i) offsets[i] = 0;
      block = dst.slice(offsets, block.dimensions());
    }

  } else {
    // Assign to block from expression.
    auto b_expr = tensor_block.expr();

    // We explicitly disable vectorization and tiling, to run a simple coefficient
    // wise assignment loop, because it's very simple and should be correct.
    using BlockAssign = TensorAssignOp<decltype(block), const decltype(b_expr)>;
    using BlockExecutor = TensorExecutor<const BlockAssign, Device, false,
                                         internal::TiledEvaluation::Off>;
    BlockExecutor::run(BlockAssign(block, b_expr), d);
  }

  // Cleanup temporary buffers owned by a tensor block.
  tensor_block.cleanup();

  // Compute a Tensor slice corresponding to a Tensor block.
  Tensor<T, NumDims, Layout> slice(block_params.desc.dimensions());
  auto s_expr = expr.slice(block_params.offsets, block_params.sizes);

  // Explicitly use coefficient assignment to evaluate slice expression.
  using SliceAssign = TensorAssignOp<decltype(slice), const decltype(s_expr)>;
  using SliceExecutor = TensorExecutor<const SliceAssign, Device, false,
                                       internal::TiledEvaluation::Off>;
  SliceExecutor::run(SliceAssign(slice, s_expr), d);

  // Tensor block and tensor slice must be the same.
  for (Index i = 0; i < block.dimensions().TotalSize(); ++i) {
    VERIFY_IS_EQUAL(block.coeff(i), slice.coeff(i));
  }
}

// -------------------------------------------------------------------------- //

template <typename T, int NumDims, int Layout>
static void test_eval_tensor_block() {
  DSizes<Index, NumDims> dims = RandomDims<NumDims>(10, 20);
  Tensor<T, NumDims, Layout> input(dims);
  input.setRandom();

  // Identity tensor expression transformation.
  VerifyBlockEvaluator<T, NumDims, Layout>(
      input, [&dims]() { return RandomBlock<Layout>(dims, 1, 10); });
}

template <typename T, int NumDims, int Layout>
static void test_eval_tensor_unary_expr_block() {
  DSizes<Index, NumDims> dims = RandomDims<NumDims>(10, 20);
  Tensor<T, NumDims, Layout> input(dims);
  input.setRandom();

  VerifyBlockEvaluator<T, NumDims, Layout>(
      input.square(), [&dims]() { return RandomBlock<Layout>(dims, 1, 10); });
}

template <typename T, int NumDims, int Layout>
static void test_eval_tensor_binary_expr_block() {
  DSizes<Index, NumDims> dims = RandomDims<NumDims>(10, 20);
  Tensor<T, NumDims, Layout> lhs(dims), rhs(dims);
  lhs.setRandom();
  rhs.setRandom();

  VerifyBlockEvaluator<T, NumDims, Layout>(
      lhs * rhs, [&dims]() { return RandomBlock<Layout>(dims, 1, 10); });
}

template <typename T, int NumDims, int Layout>
static void test_eval_tensor_binary_with_unary_expr_block() {
  DSizes<Index, NumDims> dims = RandomDims<NumDims>(10, 20);
  Tensor<T, NumDims, Layout> lhs(dims), rhs(dims);
  lhs.setRandom();
  rhs.setRandom();

  VerifyBlockEvaluator<T, NumDims, Layout>(
      (lhs.square() + rhs.square()).sqrt(),
      [&dims]() { return RandomBlock<Layout>(dims, 1, 10); });
}

template <typename T, int NumDims, int Layout>
static void test_eval_tensor_broadcast() {
  DSizes<Index, NumDims> dims = RandomDims<NumDims>(1, 10);
  Tensor<T, NumDims, Layout> input(dims);
  input.setRandom();

  DSizes<Index, NumDims> bcast = RandomDims<NumDims>(1, 5);

  DSizes<Index, NumDims> bcasted_dims;
  for (int i = 0; i < NumDims; ++i) bcasted_dims[i] = dims[i] * bcast[i];

  VerifyBlockEvaluator<T, NumDims, Layout>(
      input.broadcast(bcast),
      [&bcasted_dims]() { return SkewedInnerBlock<Layout>(bcasted_dims); });

  VerifyBlockEvaluator<T, NumDims, Layout>(
      input.broadcast(bcast),
      [&bcasted_dims]() { return RandomBlock<Layout>(bcasted_dims, 5, 10); });

  VerifyBlockEvaluator<T, NumDims, Layout>(
      input.broadcast(bcast),
      [&bcasted_dims]() { return FixedSizeBlock(bcasted_dims); });

  // Check that desc.destination() memory is not shared between two broadcast
  // materializations.
  VerifyBlockEvaluator<T, NumDims, Layout>(
      input.broadcast(bcast) * input.square().broadcast(bcast),
      [&bcasted_dims]() { return SkewedInnerBlock<Layout>(bcasted_dims); });
}

template <typename T, int NumDims, int Layout>
static void test_eval_tensor_reshape() {
  DSizes<Index, NumDims> dims = RandomDims<NumDims>(1, 10);

  DSizes<Index, NumDims> shuffled = dims;
  std::shuffle(&shuffled[0], &shuffled[NumDims - 1], std::mt19937(g_seed));

  Tensor<T, NumDims, Layout> input(dims);
  input.setRandom();

  VerifyBlockEvaluator<T, NumDims, Layout>(
      input.reshape(shuffled),
      [&shuffled]() { return RandomBlock<Layout>(shuffled, 1, 10); });

  VerifyBlockEvaluator<T, NumDims, Layout>(
      input.reshape(shuffled),
      [&shuffled]() { return SkewedInnerBlock<Layout>(shuffled); });
}

template <typename T, int NumDims, int Layout>
static void test_eval_tensor_cast() {
  DSizes<Index, NumDims> dims = RandomDims<NumDims>(10, 20);
  Tensor<T, NumDims, Layout> input(dims);
  input.setRandom();

  VerifyBlockEvaluator<T, NumDims, Layout>(
      input.template cast<int>().template cast<T>(),
      [&dims]() { return RandomBlock<Layout>(dims, 1, 10); });
}

template <typename T, int NumDims, int Layout>
static void test_eval_tensor_select() {
  DSizes<Index, NumDims> dims = RandomDims<NumDims>(10, 20);
  Tensor<T, NumDims, Layout> lhs(dims);
  Tensor<T, NumDims, Layout> rhs(dims);
  Tensor<bool, NumDims, Layout> cond(dims);
  lhs.setRandom();
  rhs.setRandom();
  cond.setRandom();

  VerifyBlockEvaluator<T, NumDims, Layout>(cond.select(lhs, rhs), [&dims]() {
    return RandomBlock<Layout>(dims, 1, 20);
  });
}

template <typename T, int NumDims, int Layout>
static void test_eval_tensor_padding() {
  const int inner_dim = Layout == static_cast<int>(ColMajor) ? 0 : NumDims - 1;

  DSizes<Index, NumDims> dims = RandomDims<NumDims>(10, 20);
  Tensor<T, NumDims, Layout> input(dims);
  input.setRandom();

  DSizes<Index, NumDims> pad_before = RandomDims<NumDims>(0, 4);
  DSizes<Index, NumDims> pad_after = RandomDims<NumDims>(0, 4);
  array<std::pair<Index, Index>, NumDims> paddings;
  for (int i = 0; i < NumDims; ++i) {
    paddings[i] = std::make_pair(pad_before[i], pad_after[i]);
  }

  // Test squeezing reads from inner dim.
  if (internal::random<bool>()) {
    pad_before[inner_dim] = 0;
    pad_after[inner_dim] = 0;
    paddings[inner_dim] = std::make_pair(0, 0);
  }

  DSizes<Index, NumDims> padded_dims;
  for (int i = 0; i < NumDims; ++i) {
    padded_dims[i] = dims[i] + pad_before[i] + pad_after[i];
  }

  VerifyBlockEvaluator<T, NumDims, Layout>(
      input.pad(paddings),
      [&padded_dims]() { return FixedSizeBlock(padded_dims); });

  VerifyBlockEvaluator<T, NumDims, Layout>(
      input.pad(paddings),
      [&padded_dims]() { return RandomBlock<Layout>(padded_dims, 1, 10); });

  VerifyBlockEvaluator<T, NumDims, Layout>(
      input.pad(paddings),
      [&padded_dims]() { return SkewedInnerBlock<Layout>(padded_dims); });
}

template <typename T, int NumDims, int Layout>
static void test_eval_tensor_chipping() {
  DSizes<Index, NumDims> dims = RandomDims<NumDims>(10, 20);
  Tensor<T, NumDims, Layout> input(dims);
  input.setRandom();

  Index chip_dim = internal::random<int>(0, NumDims - 1);
  Index chip_offset = internal::random<Index>(0, dims[chip_dim] - 2);

  DSizes<Index, NumDims - 1> chipped_dims;
  for (Index i = 0; i < chip_dim; ++i) {
    chipped_dims[i] = dims[i];
  }
  for (Index i = chip_dim + 1; i < NumDims; ++i) {
    chipped_dims[i - 1] = dims[i];
  }

  // Block buffer forwarding.
  VerifyBlockEvaluator<T, NumDims - 1, Layout>(
      input.chip(chip_offset, chip_dim),
      [&chipped_dims]() { return FixedSizeBlock(chipped_dims); });

  VerifyBlockEvaluator<T, NumDims - 1, Layout>(
      input.chip(chip_offset, chip_dim),
      [&chipped_dims]() { return RandomBlock<Layout>(chipped_dims, 1, 10); });

  // Block expression assignment.
  VerifyBlockEvaluator<T, NumDims - 1, Layout>(
      input.square().chip(chip_offset, chip_dim),
      [&chipped_dims]() { return FixedSizeBlock(chipped_dims); });

  VerifyBlockEvaluator<T, NumDims - 1, Layout>(
      input.square().chip(chip_offset, chip_dim),
      [&chipped_dims]() { return RandomBlock<Layout>(chipped_dims, 1, 10); });
}

template <typename T, int NumDims, int Layout>
static void test_eval_tensor_generator() {
  DSizes<Index, NumDims> dims = RandomDims<NumDims>(10, 20);
  Tensor<T, NumDims, Layout> input(dims);
  input.setRandom();

  auto generator = [](const array<Index, NumDims>& coords) -> T {
    T result = static_cast<T>(0);
    for (int i = 0; i < NumDims; ++i) {
      result += static_cast<T>((i + 1) * coords[i]);
    }
    return result;
  };

  VerifyBlockEvaluator<T, NumDims, Layout>(
      input.generate(generator), [&dims]() { return FixedSizeBlock(dims); });

  VerifyBlockEvaluator<T, NumDims, Layout>(
      input.generate(generator),
      [&dims]() { return RandomBlock<Layout>(dims, 1, 10); });
}

template <typename T, int NumDims, int Layout>
static void test_eval_tensor_reverse() {
  DSizes<Index, NumDims> dims = RandomDims<NumDims>(10, 20);
  Tensor<T, NumDims, Layout> input(dims);
  input.setRandom();

  // Randomly reverse dimensions.
  Eigen::DSizes<bool, NumDims> reverse;
  for (int i = 0; i < NumDims; ++i) reverse[i] = internal::random<bool>();

  VerifyBlockEvaluator<T, NumDims, Layout>(
      input.reverse(reverse), [&dims]() { return FixedSizeBlock(dims); });

  VerifyBlockEvaluator<T, NumDims, Layout>(input.reverse(reverse), [&dims]() {
    return RandomBlock<Layout>(dims, 1, 10);
  });
}

template <typename T, int NumDims, int Layout>
static void test_eval_tensor_slice() {
  DSizes<Index, NumDims> dims = RandomDims<NumDims>(10, 20);
  Tensor<T, NumDims, Layout> input(dims);
  input.setRandom();

  // Pick a random slice of an input tensor.
  DSizes<Index, NumDims> slice_start = RandomDims<NumDims>(5, 10);
  DSizes<Index, NumDims> slice_size = RandomDims<NumDims>(5, 10);

  // Make sure that slice start + size do not overflow tensor dims.
  for (int i = 0; i < NumDims; ++i) {
    slice_start[i] = numext::mini(dims[i] - 1, slice_start[i]);
    slice_size[i] = numext::mini(slice_size[i], dims[i] - slice_start[i]);
  }

  VerifyBlockEvaluator<T, NumDims, Layout>(
      input.slice(slice_start, slice_size),
      [&slice_size]() { return FixedSizeBlock(slice_size); });

  VerifyBlockEvaluator<T, NumDims, Layout>(
      input.slice(slice_start, slice_size),
      [&slice_size]() { return RandomBlock<Layout>(slice_size, 1, 10); });
}

template <typename T, int NumDims, int Layout>
static void test_eval_tensor_shuffle() {
  DSizes<Index, NumDims> dims = RandomDims<NumDims>(5, 15);
  Tensor<T, NumDims, Layout> input(dims);
  input.setRandom();

  DSizes<Index, NumDims> shuffle;
  for (int i = 0; i < NumDims; ++i) shuffle[i] = i;

  do {
    DSizes<Index, NumDims> shuffled_dims;
    for (int i = 0; i < NumDims; ++i) shuffled_dims[i] = dims[shuffle[i]];

    VerifyBlockEvaluator<T, NumDims, Layout>(
        input.shuffle(shuffle),
        [&shuffled_dims]() { return FixedSizeBlock(shuffled_dims); });

    VerifyBlockEvaluator<T, NumDims, Layout>(
        input.shuffle(shuffle), [&shuffled_dims]() {
          return RandomBlock<Layout>(shuffled_dims, 1, 5);
        });

    break;

  } while (std::next_permutation(&shuffle[0], &shuffle[0] + NumDims));
}

template <typename T, int Layout>
static void test_eval_tensor_reshape_with_bcast() {
  Index dim = internal::random<Index>(1, 100);

  Tensor<T, 2, Layout> lhs(1, dim);
  Tensor<T, 2, Layout> rhs(dim, 1);
  lhs.setRandom();
  rhs.setRandom();

  auto reshapeLhs = NByOne(dim);
  auto reshapeRhs = OneByM(dim);

  auto bcastLhs = OneByM(dim);
  auto bcastRhs = NByOne(dim);

  DSizes<Index, 2> dims(dim, dim);

  VerifyBlockEvaluator<T, 2, Layout>(
      lhs.reshape(reshapeLhs).broadcast(bcastLhs) *
          rhs.reshape(reshapeRhs).broadcast(bcastRhs),
      [dims]() { return SkewedInnerBlock<Layout, 2>(dims); });
}

template <typename T, int Layout>
static void test_eval_tensor_forced_eval() {
  Index dim = internal::random<Index>(1, 100);

  Tensor<T, 2, Layout> lhs(dim, 1);
  Tensor<T, 2, Layout> rhs(1, dim);
  lhs.setRandom();
  rhs.setRandom();

  auto bcastLhs = OneByM(dim);
  auto bcastRhs = NByOne(dim);

  DSizes<Index, 2> dims(dim, dim);

  VerifyBlockEvaluator<T, 2, Layout>(
      (lhs.broadcast(bcastLhs) * rhs.broadcast(bcastRhs)).eval().reshape(dims),
      [dims]() { return SkewedInnerBlock<Layout, 2>(dims); });

  VerifyBlockEvaluator<T, 2, Layout>(
      (lhs.broadcast(bcastLhs) * rhs.broadcast(bcastRhs)).eval().reshape(dims),
      [dims]() { return RandomBlock<Layout, 2>(dims, 1, 50); });
}

template <typename T, int Layout>
static void test_eval_tensor_chipping_of_bcast() {
  if (Layout != static_cast<int>(RowMajor)) return;

  Index dim0 = internal::random<Index>(1, 10);
  Index dim1 = internal::random<Index>(1, 10);
  Index dim2 = internal::random<Index>(1, 10);

  Tensor<T, 3, Layout> input(1, dim1, dim2);
  input.setRandom();

  Eigen::array<Index, 3> bcast = {{dim0, 1, 1}};
  DSizes<Index, 2> chipped_dims(dim0, dim2);

  VerifyBlockEvaluator<T, 2, Layout>(
      input.broadcast(bcast).chip(0, 1),
      [chipped_dims]() { return FixedSizeBlock(chipped_dims); });

  VerifyBlockEvaluator<T, 2, Layout>(
      input.broadcast(bcast).chip(0, 1),
      [chipped_dims]() { return SkewedInnerBlock<Layout, 2>(chipped_dims); });

  VerifyBlockEvaluator<T, 2, Layout>(
      input.broadcast(bcast).chip(0, 1),
      [chipped_dims]() { return RandomBlock<Layout, 2>(chipped_dims, 1, 5); });
}

// -------------------------------------------------------------------------- //
// Verify that assigning block to a Tensor expression produces the same result
// as an assignment to TensorSliceOp (writing a block is is identical to
// assigning one tensor to a slice of another tensor).

template <typename T, int NumDims, int Layout, int NumExprDims = NumDims,
          typename Expression, typename GenBlockParams>
static void VerifyBlockAssignment(Tensor<T, NumDims, Layout>& tensor,
                                  Expression expr, GenBlockParams gen_block) {
  using Device = DefaultDevice;
  auto d = Device();

  // We use tensor evaluator as a target for block and slice assignments.
  auto eval = TensorEvaluator<decltype(expr), Device>(expr, d);

  // Generate a random block, or choose a block that fits in full expression.
  TensorBlockParams<NumExprDims> block_params = gen_block();

  // Generate random data of the selected block size.
  Tensor<T, NumExprDims, Layout> block(block_params.desc.dimensions());
  block.setRandom();

  // ************************************************************************ //
  // (1) Assignment from a block.

  // Construct a materialize block from a random generated block tensor.
  internal::TensorMaterializedBlock<T, NumExprDims, Layout> blk(
      internal::TensorBlockKind::kView, block.data(), block.dimensions());

  // Reset all underlying tensor values to zero.
  tensor.setZero();

  // Use evaluator to write block into a tensor.
  eval.writeBlock(block_params.desc, blk);

  // Make a copy of the result after assignment.
  Tensor<T, NumDims, Layout> block_assigned = tensor;

  // ************************************************************************ //
  // (2) Assignment to a slice

  // Reset all underlying tensor values to zero.
  tensor.setZero();

  // Assign block to a slice of original expression
  auto s_expr = expr.slice(block_params.offsets, block_params.sizes);

  // Explicitly use coefficient assignment to evaluate slice expression.
  using SliceAssign = TensorAssignOp<decltype(s_expr), const decltype(block)>;
  using SliceExecutor = TensorExecutor<const SliceAssign, Device, false,
                                       internal::TiledEvaluation::Off>;
  SliceExecutor::run(SliceAssign(s_expr, block), d);

  // Make a copy of the result after assignment.
  Tensor<T, NumDims, Layout> slice_assigned = tensor;

  for (Index i = 0; i < tensor.dimensions().TotalSize(); ++i) {
    VERIFY_IS_EQUAL(block_assigned.coeff(i), slice_assigned.coeff(i));
  }
}

// -------------------------------------------------------------------------- //

template <typename T, int NumDims, int Layout>
static void test_assign_to_tensor() {
  DSizes<Index, NumDims> dims = RandomDims<NumDims>(10, 20);
  Tensor<T, NumDims, Layout> tensor(dims);

  TensorMap<Tensor<T, NumDims, Layout>> map(tensor.data(), dims);

  VerifyBlockAssignment<T, NumDims, Layout>(
      tensor, map, [&dims]() { return RandomBlock<Layout>(dims, 10, 20); });
  VerifyBlockAssignment<T, NumDims, Layout>(
      tensor, map, [&dims]() { return FixedSizeBlock(dims); });
}

template <typename T, int NumDims, int Layout>
static void test_assign_to_tensor_reshape() {
  DSizes<Index, NumDims> dims = RandomDims<NumDims>(10, 20);
  Tensor<T, NumDims, Layout> tensor(dims);

  TensorMap<Tensor<T, NumDims, Layout>> map(tensor.data(), dims);

  DSizes<Index, NumDims> shuffled = dims;
  std::shuffle(&shuffled[0], &shuffled[NumDims - 1], std::mt19937(g_seed));

  VerifyBlockAssignment<T, NumDims, Layout>(
      tensor, map.reshape(shuffled),
      [&shuffled]() { return RandomBlock<Layout>(shuffled, 1, 10); });

  VerifyBlockAssignment<T, NumDims, Layout>(
      tensor, map.reshape(shuffled),
      [&shuffled]() { return SkewedInnerBlock<Layout>(shuffled); });

  VerifyBlockAssignment<T, NumDims, Layout>(
      tensor, map.reshape(shuffled),
      [&shuffled]() { return FixedSizeBlock(shuffled); });
}

template <typename T, int NumDims, int Layout>
static void test_assign_to_tensor_chipping() {
  DSizes<Index, NumDims> dims = RandomDims<NumDims>(10, 20);
  Tensor<T, NumDims, Layout> tensor(dims);

  Index chip_dim = internal::random<int>(0, NumDims - 1);
  Index chip_offset = internal::random<Index>(0, dims[chip_dim] - 2);

  DSizes<Index, NumDims - 1> chipped_dims;
  for (Index i = 0; i < chip_dim; ++i) {
    chipped_dims[i] = dims[i];
  }
  for (Index i = chip_dim + 1; i < NumDims; ++i) {
    chipped_dims[i - 1] = dims[i];
  }

  TensorMap<Tensor<T, NumDims, Layout>> map(tensor.data(), dims);

  VerifyBlockAssignment<T, NumDims, Layout, NumDims - 1>(
      tensor, map.chip(chip_offset, chip_dim),
      [&chipped_dims]() { return RandomBlock<Layout>(chipped_dims, 1, 10); });

  VerifyBlockAssignment<T, NumDims, Layout, NumDims - 1>(
      tensor, map.chip(chip_offset, chip_dim),
      [&chipped_dims]() { return SkewedInnerBlock<Layout>(chipped_dims); });

  VerifyBlockAssignment<T, NumDims, Layout, NumDims - 1>(
      tensor, map.chip(chip_offset, chip_dim),
      [&chipped_dims]() { return FixedSizeBlock(chipped_dims); });
}

template <typename T, int NumDims, int Layout>
static void test_assign_to_tensor_slice() {
  DSizes<Index, NumDims> dims = RandomDims<NumDims>(10, 20);
  Tensor<T, NumDims, Layout> tensor(dims);

  // Pick a random slice of tensor.
  DSizes<Index, NumDims> slice_start = RandomDims<NumDims>(5, 10);
  DSizes<Index, NumDims> slice_size = RandomDims<NumDims>(5, 10);

  // Make sure that slice start + size do not overflow tensor dims.
  for (int i = 0; i < NumDims; ++i) {
    slice_start[i] = numext::mini(dims[i] - 1, slice_start[i]);
    slice_size[i] = numext::mini(slice_size[i], dims[i] - slice_start[i]);
  }

  TensorMap<Tensor<T, NumDims, Layout>> map(tensor.data(), dims);

  VerifyBlockAssignment<T, NumDims, Layout>(
      tensor, map.slice(slice_start, slice_size),
      [&slice_size]() { return RandomBlock<Layout>(slice_size, 1, 10); });

  VerifyBlockAssignment<T, NumDims, Layout>(
      tensor, map.slice(slice_start, slice_size),
      [&slice_size]() { return SkewedInnerBlock<Layout>(slice_size); });

  VerifyBlockAssignment<T, NumDims, Layout>(
      tensor, map.slice(slice_start, slice_size),
      [&slice_size]() { return FixedSizeBlock(slice_size); });
}

template <typename T, int NumDims, int Layout>
static void test_assign_to_tensor_shuffle() {
  DSizes<Index, NumDims> dims = RandomDims<NumDims>(5, 15);
  Tensor<T, NumDims, Layout> tensor(dims);

  DSizes<Index, NumDims> shuffle;
  for (int i = 0; i < NumDims; ++i) shuffle[i] = i;

  TensorMap<Tensor<T, NumDims, Layout>> map(tensor.data(), dims);

  do {
    DSizes<Index, NumDims> shuffled_dims;
    for (int i = 0; i < NumDims; ++i) shuffled_dims[i] = dims[shuffle[i]];

    VerifyBlockAssignment<T, NumDims, Layout>(
        tensor, map.shuffle(shuffle),
        [&shuffled_dims]() { return FixedSizeBlock(shuffled_dims); });

    VerifyBlockAssignment<T, NumDims, Layout>(
        tensor, map.shuffle(shuffle), [&shuffled_dims]() {
          return RandomBlock<Layout>(shuffled_dims, 1, 5);
        });

  } while (std::next_permutation(&shuffle[0], &shuffle[0] + NumDims));
}

// -------------------------------------------------------------------------- //

#define CALL_SUBTEST_PART(PART) \
  CALL_SUBTEST_##PART

#define CALL_SUBTESTS_DIMS_LAYOUTS_TYPES(PART, NAME)           \
  CALL_SUBTEST_PART(PART)((NAME<float, 1, RowMajor>())); \
  CALL_SUBTEST_PART(PART)((NAME<float, 2, RowMajor>())); \
  CALL_SUBTEST_PART(PART)((NAME<float, 3, RowMajor>())); \
  CALL_SUBTEST_PART(PART)((NAME<float, 4, RowMajor>())); \
  CALL_SUBTEST_PART(PART)((NAME<float, 5, RowMajor>())); \
  CALL_SUBTEST_PART(PART)((NAME<float, 1, ColMajor>())); \
  CALL_SUBTEST_PART(PART)((NAME<float, 2, ColMajor>())); \
  CALL_SUBTEST_PART(PART)((NAME<float, 4, ColMajor>())); \
  CALL_SUBTEST_PART(PART)((NAME<float, 4, ColMajor>())); \
  CALL_SUBTEST_PART(PART)((NAME<float, 5, ColMajor>())); \
  CALL_SUBTEST_PART(PART)((NAME<int, 1, RowMajor>())); \
  CALL_SUBTEST_PART(PART)((NAME<int, 2, RowMajor>())); \
  CALL_SUBTEST_PART(PART)((NAME<int, 3, RowMajor>())); \
  CALL_SUBTEST_PART(PART)((NAME<int, 4, RowMajor>())); \
  CALL_SUBTEST_PART(PART)((NAME<int, 5, RowMajor>())); \
  CALL_SUBTEST_PART(PART)((NAME<int, 1, ColMajor>())); \
  CALL_SUBTEST_PART(PART)((NAME<int, 2, ColMajor>())); \
  CALL_SUBTEST_PART(PART)((NAME<int, 4, ColMajor>())); \
  CALL_SUBTEST_PART(PART)((NAME<int, 4, ColMajor>())); \
  CALL_SUBTEST_PART(PART)((NAME<int, 5, ColMajor>())); \
  CALL_SUBTEST_PART(PART)((NAME<bool, 1, RowMajor>())); \
  CALL_SUBTEST_PART(PART)((NAME<bool, 2, RowMajor>())); \
  CALL_SUBTEST_PART(PART)((NAME<bool, 3, RowMajor>())); \
  CALL_SUBTEST_PART(PART)((NAME<bool, 4, RowMajor>())); \
  CALL_SUBTEST_PART(PART)((NAME<bool, 5, RowMajor>())); \
  CALL_SUBTEST_PART(PART)((NAME<bool, 1, ColMajor>())); \
  CALL_SUBTEST_PART(PART)((NAME<bool, 2, ColMajor>())); \
  CALL_SUBTEST_PART(PART)((NAME<bool, 4, ColMajor>())); \
  CALL_SUBTEST_PART(PART)((NAME<bool, 4, ColMajor>())); \
  CALL_SUBTEST_PART(PART)((NAME<bool, 5, ColMajor>()))

#define CALL_SUBTESTS_DIMS_LAYOUTS(PART, NAME)     \
  CALL_SUBTEST_PART(PART)((NAME<float, 1, RowMajor>())); \
  CALL_SUBTEST_PART(PART)((NAME<float, 2, RowMajor>())); \
  CALL_SUBTEST_PART(PART)((NAME<float, 3, RowMajor>())); \
  CALL_SUBTEST_PART(PART)((NAME<float, 4, RowMajor>())); \
  CALL_SUBTEST_PART(PART)((NAME<float, 5, RowMajor>())); \
  CALL_SUBTEST_PART(PART)((NAME<float, 1, ColMajor>())); \
  CALL_SUBTEST_PART(PART)((NAME<float, 2, ColMajor>())); \
  CALL_SUBTEST_PART(PART)((NAME<float, 4, ColMajor>())); \
  CALL_SUBTEST_PART(PART)((NAME<float, 4, ColMajor>())); \
  CALL_SUBTEST_PART(PART)((NAME<float, 5, ColMajor>()))

#define CALL_SUBTESTS_LAYOUTS_TYPES(PART, NAME)       \
  CALL_SUBTEST_PART(PART)((NAME<float, RowMajor>())); \
  CALL_SUBTEST_PART(PART)((NAME<float, ColMajor>()));  \
  CALL_SUBTEST_PART(PART)((NAME<bool, RowMajor>())); \
  CALL_SUBTEST_PART(PART)((NAME<bool, ColMajor>()))

EIGEN_DECLARE_TEST(cxx11_tensor_block_eval) {
  // clang-format off
  CALL_SUBTESTS_DIMS_LAYOUTS_TYPES(1, test_eval_tensor_block);
  CALL_SUBTESTS_DIMS_LAYOUTS_TYPES(1, test_eval_tensor_binary_expr_block);
  CALL_SUBTESTS_DIMS_LAYOUTS(1, test_eval_tensor_unary_expr_block);
  CALL_SUBTESTS_DIMS_LAYOUTS(2, test_eval_tensor_binary_with_unary_expr_block);
  CALL_SUBTESTS_DIMS_LAYOUTS_TYPES(2, test_eval_tensor_broadcast);
  CALL_SUBTESTS_DIMS_LAYOUTS_TYPES(2, test_eval_tensor_reshape);
  CALL_SUBTESTS_DIMS_LAYOUTS_TYPES(3, test_eval_tensor_cast);
  CALL_SUBTESTS_DIMS_LAYOUTS_TYPES(3, test_eval_tensor_select);
  CALL_SUBTESTS_DIMS_LAYOUTS_TYPES(3, test_eval_tensor_padding);
  CALL_SUBTESTS_DIMS_LAYOUTS_TYPES(4, test_eval_tensor_chipping);
  CALL_SUBTESTS_DIMS_LAYOUTS_TYPES(4, test_eval_tensor_generator);
  CALL_SUBTESTS_DIMS_LAYOUTS_TYPES(4, test_eval_tensor_reverse);
  CALL_SUBTESTS_DIMS_LAYOUTS_TYPES(5, test_eval_tensor_slice);
  CALL_SUBTESTS_DIMS_LAYOUTS_TYPES(5, test_eval_tensor_shuffle);

  CALL_SUBTESTS_LAYOUTS_TYPES(6, test_eval_tensor_reshape_with_bcast);
  CALL_SUBTESTS_LAYOUTS_TYPES(6, test_eval_tensor_forced_eval);
  CALL_SUBTESTS_LAYOUTS_TYPES(6, test_eval_tensor_chipping_of_bcast);

  CALL_SUBTESTS_DIMS_LAYOUTS_TYPES(7, test_assign_to_tensor);
  CALL_SUBTESTS_DIMS_LAYOUTS_TYPES(7, test_assign_to_tensor_reshape);
  CALL_SUBTESTS_DIMS_LAYOUTS_TYPES(7, test_assign_to_tensor_chipping);
  CALL_SUBTESTS_DIMS_LAYOUTS_TYPES(8, test_assign_to_tensor_slice);
  CALL_SUBTESTS_DIMS_LAYOUTS_TYPES(8, test_assign_to_tensor_shuffle);

  // Force CMake to split this test.
  // EIGEN_SUFFIXES;1;2;3;4;5;6;7;8

  // clang-format on
}
