// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2018 Eugene Zhulenev <ezhulenev@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_USE_THREADS

#include "main.h"

#include <Eigen/CXX11/Tensor>

using Eigen::Tensor;
using Eigen::RowMajor;
using Eigen::ColMajor;
using Eigen::internal::TiledEvaluation;

// A set of tests to verify that different TensorExecutor strategies yields the
// same results for all the ops, supporting tiled evaluation.

// Default assignment that does no use block evaluation or vectorization.
// We assume that default coefficient evaluation is well tested and correct.
template <typename Dst, typename Expr>
static void DefaultAssign(Dst& dst, Expr expr) {
  using Assign = Eigen::TensorAssignOp<Dst, const Expr>;
  using Executor =
      Eigen::internal::TensorExecutor<const Assign, DefaultDevice,
                                      /*Vectorizable=*/false,
                                      /*Tiling=*/TiledEvaluation::Off>;

  Executor::run(Assign(dst, expr), DefaultDevice());
}

// Assignment with specified device and tiling strategy.
template <bool Vectorizable, TiledEvaluation Tiling, typename Device,
          typename Dst, typename Expr>
static void DeviceAssign(Device& d, Dst& dst, Expr expr) {
  using Assign = Eigen::TensorAssignOp<Dst, const Expr>;
  using Executor = Eigen::internal::TensorExecutor<const Assign, Device,
                                                   Vectorizable, Tiling>;

  Executor::run(Assign(dst, expr), d);
}

template <int NumDims>
static array<Index, NumDims> RandomDims(int min_dim = 1, int max_dim = 20) {
  array<Index, NumDims> dims;
  for (int i = 0; i < NumDims; ++i) {
    dims[i] = internal::random<int>(min_dim, max_dim);
  }
  return dims;
}

template <typename T, int NumDims, typename Device, bool Vectorizable,
          TiledEvaluation Tiling, int Layout>
static void test_execute_unary_expr(Device d)
{
  static constexpr int Options = 0 | Layout;

  // Pick a large enough tensor size to bypass small tensor block evaluation
  // optimization.
  auto dims = RandomDims<NumDims>(50 / NumDims, 100 / NumDims);

  Tensor<T, NumDims, Options, Index> src(dims);
  Tensor<T, NumDims, Options, Index> dst(dims);

  src.setRandom();
  const auto expr = src.square();

  using Assign = TensorAssignOp<decltype(dst), const decltype(expr)>;
  using Executor =
      internal::TensorExecutor<const Assign, Device, Vectorizable, Tiling>;

  Executor::run(Assign(dst, expr), d);

  for (Index i = 0; i < dst.dimensions().TotalSize(); ++i) {
    T square = src.coeff(i) * src.coeff(i);
    VERIFY_IS_EQUAL(square, dst.coeff(i));
  }
}

template <typename T, int NumDims, typename Device, bool Vectorizable,
          TiledEvaluation Tiling, int Layout>
static void test_execute_binary_expr(Device d)
{
  static constexpr int Options = 0 | Layout;

  // Pick a large enough tensor size to bypass small tensor block evaluation
  // optimization.
  auto dims = RandomDims<NumDims>(50 / NumDims, 100 / NumDims);

  Tensor<T, NumDims, Options, Index> lhs(dims);
  Tensor<T, NumDims, Options, Index> rhs(dims);
  Tensor<T, NumDims, Options, Index> dst(dims);

  lhs.setRandom();
  rhs.setRandom();

  const auto expr = lhs + rhs;

  using Assign = TensorAssignOp<decltype(dst), const decltype(expr)>;
  using Executor =
      internal::TensorExecutor<const Assign, Device, Vectorizable, Tiling>;

  Executor::run(Assign(dst, expr), d);

  for (Index i = 0; i < dst.dimensions().TotalSize(); ++i) {
    T sum = lhs.coeff(i) + rhs.coeff(i);
    VERIFY_IS_EQUAL(sum, dst.coeff(i));
  }
}

template <typename T, int NumDims, typename Device, bool Vectorizable,
          TiledEvaluation Tiling, int Layout>
static void test_execute_broadcasting(Device d)
{
  static constexpr int Options = 0 | Layout;

  auto dims = RandomDims<NumDims>(1, 10);
  Tensor<T, NumDims, Options, Index> src(dims);
  src.setRandom();

  const auto broadcasts = RandomDims<NumDims>(1, 7);
  const auto expr = src.broadcast(broadcasts);

  // We assume that broadcasting on a default device is tested and correct, so
  // we can rely on it to verify correctness of tensor executor and tiling.
  Tensor<T, NumDims, Options, Index> golden;
  golden = expr;

  // Now do the broadcasting using configured tensor executor.
  Tensor<T, NumDims, Options, Index> dst(golden.dimensions());

  using Assign = TensorAssignOp<decltype(dst), const decltype(expr)>;
  using Executor =
      internal::TensorExecutor<const Assign, Device, Vectorizable, Tiling>;

  Executor::run(Assign(dst, expr), d);

  for (Index i = 0; i < dst.dimensions().TotalSize(); ++i) {
    VERIFY_IS_EQUAL(dst.coeff(i), golden.coeff(i));
  }
}

template <typename T, int NumDims, typename Device, bool Vectorizable,
          TiledEvaluation Tiling, int Layout>
static void test_execute_chipping_rvalue(Device d)
{
  auto dims = RandomDims<NumDims>(1, 10);
  Tensor<T, NumDims, Layout, Index> src(dims);
  src.setRandom();

#define TEST_CHIPPING(CHIP_DIM)                                           \
  if (NumDims > (CHIP_DIM)) {                                             \
    const auto offset = internal::random<Index>(0, dims[(CHIP_DIM)] - 1); \
    const auto expr = src.template chip<(CHIP_DIM)>(offset);              \
                                                                          \
    Tensor<T, NumDims - 1, Layout, Index> golden;                         \
    golden = expr;                                                        \
                                                                          \
    Tensor<T, NumDims - 1, Layout, Index> dst(golden.dimensions());       \
                                                                          \
    using Assign = TensorAssignOp<decltype(dst), const decltype(expr)>;   \
    using Executor = internal::TensorExecutor<const Assign, Device,       \
                                              Vectorizable, Tiling>;      \
                                                                          \
    Executor::run(Assign(dst, expr), d);                                  \
                                                                          \
    for (Index i = 0; i < dst.dimensions().TotalSize(); ++i) {            \
      VERIFY_IS_EQUAL(dst.coeff(i), golden.coeff(i));                     \
    }                                                                     \
  }

  TEST_CHIPPING(0)
  TEST_CHIPPING(1)
  TEST_CHIPPING(2)
  TEST_CHIPPING(3)
  TEST_CHIPPING(4)
  TEST_CHIPPING(5)

#undef TEST_CHIPPING
}

template <typename T, int NumDims, typename Device, bool Vectorizable,
    TiledEvaluation Tiling, int Layout>
static void test_execute_chipping_lvalue(Device d)
{
  auto dims = RandomDims<NumDims>(1, 10);

#define TEST_CHIPPING(CHIP_DIM)                                             \
  if (NumDims > (CHIP_DIM)) {                                               \
    /* Generate random data that we'll assign to the chipped tensor dim. */ \
    array<Index, NumDims - 1> src_dims;                                     \
    for (int i = 0; i < NumDims - 1; ++i) {                                 \
      int dim = i < (CHIP_DIM) ? i : i + 1;                                 \
      src_dims[i] = dims[dim];                                              \
    }                                                                       \
                                                                            \
    Tensor<T, NumDims - 1, Layout, Index> src(src_dims);                    \
    src.setRandom();                                                        \
                                                                            \
    const auto offset = internal::random<Index>(0, dims[(CHIP_DIM)] - 1);   \
                                                                            \
    Tensor<T, NumDims, Layout, Index> random(dims);                         \
    random.setZero();                                                       \
                                                                            \
    Tensor<T, NumDims, Layout, Index> golden(dims);                         \
    golden = random;                                                        \
    golden.template chip<(CHIP_DIM)>(offset) = src;                         \
                                                                            \
    Tensor<T, NumDims, Layout, Index> dst(dims);                            \
    dst = random;                                                           \
    auto expr = dst.template chip<(CHIP_DIM)>(offset);                      \
                                                                            \
    using Assign = TensorAssignOp<decltype(expr), const decltype(src)>;     \
    using Executor = internal::TensorExecutor<const Assign, Device,         \
                                              Vectorizable, Tiling>;        \
                                                                            \
    Executor::run(Assign(expr, src), d);                                    \
                                                                            \
    for (Index i = 0; i < dst.dimensions().TotalSize(); ++i) {              \
      VERIFY_IS_EQUAL(dst.coeff(i), golden.coeff(i));                       \
    }                                                                       \
  }

  TEST_CHIPPING(0)
  TEST_CHIPPING(1)
  TEST_CHIPPING(2)
  TEST_CHIPPING(3)
  TEST_CHIPPING(4)
  TEST_CHIPPING(5)

#undef TEST_CHIPPING
}

template <typename T, int NumDims, typename Device, bool Vectorizable,
          TiledEvaluation Tiling, int Layout>
static void test_execute_shuffle_rvalue(Device d)
{
  static constexpr int Options = 0 | Layout;

  auto dims = RandomDims<NumDims>(1, 10);
  Tensor<T, NumDims, Options, Index> src(dims);
  src.setRandom();

  DSizes<Index, NumDims> shuffle;
  for (int i = 0; i < NumDims; ++i) shuffle[i] = i;

  // Test all possible shuffle permutations.
  do {
    DSizes<Index, NumDims> shuffled_dims;
    for (int i = 0; i < NumDims; ++i) {
      shuffled_dims[i] = dims[shuffle[i]];
    }

    const auto expr = src.shuffle(shuffle);

    // We assume that shuffling on a default device is tested and correct, so
    // we can rely on it to verify correctness of tensor executor and tiling.
    Tensor<T, NumDims, Options, Index> golden(shuffled_dims);
    DefaultAssign(golden, expr);

    // Now do the shuffling using configured tensor executor.
    Tensor<T, NumDims, Options, Index> dst(shuffled_dims);
    DeviceAssign<Vectorizable, Tiling>(d, dst, expr);

    for (Index i = 0; i < dst.dimensions().TotalSize(); ++i) {
      VERIFY_IS_EQUAL(dst.coeff(i), golden.coeff(i));
    }

  } while (std::next_permutation(&shuffle[0], &shuffle[0] + NumDims));
}

template <typename T, int NumDims, typename Device, bool Vectorizable,
          TiledEvaluation Tiling, int Layout>
static void test_execute_shuffle_lvalue(Device d)
{
  static constexpr int Options = 0 | Layout;

  auto dims = RandomDims<NumDims>(5, 10);
  Tensor<T, NumDims, Options, Index> src(dims);
  src.setRandom();

  DSizes<Index, NumDims> shuffle;
  for (int i = 0; i < NumDims; ++i) shuffle[i] = i;

  // Test all possible shuffle permutations.
  do {
    DSizes<Index, NumDims> shuffled_dims;
    for (int i = 0; i < NumDims; ++i) shuffled_dims[shuffle[i]] = dims[i];

    // We assume that shuffling on a default device is tested and correct, so
    // we can rely on it to verify correctness of tensor executor and tiling.
    Tensor<T, NumDims, Options, Index> golden(shuffled_dims);
    auto golden_shuffle = golden.shuffle(shuffle);
    DefaultAssign(golden_shuffle, src);

    // Now do the shuffling using configured tensor executor.
    Tensor<T, NumDims, Options, Index> dst(shuffled_dims);
    auto dst_shuffle = dst.shuffle(shuffle);
    DeviceAssign<Vectorizable, Tiling>(d, dst_shuffle, src);

    for (Index i = 0; i < dst.dimensions().TotalSize(); ++i) {
      VERIFY_IS_EQUAL(dst.coeff(i), golden.coeff(i));
    }

  } while (std::next_permutation(&shuffle[0], &shuffle[0] + NumDims));
}

template <typename T, int NumDims, typename Device, bool Vectorizable,
    TiledEvaluation Tiling, int Layout>
static void test_execute_reshape(Device d)
{
  static_assert(NumDims >= 2, "NumDims must be greater or equal than 2");

  static constexpr int ReshapedDims = NumDims - 1;
  static constexpr int Options = 0 | Layout;

  auto dims = RandomDims<NumDims>(5, 10);
  Tensor<T, NumDims, Options, Index> src(dims);
  src.setRandom();

  // Multiple 0th dimension and then shuffle.
  std::vector<Index> shuffle;
  for (int i = 0; i < ReshapedDims; ++i) shuffle.push_back(i);
  std::shuffle(shuffle.begin(), shuffle.end(), std::mt19937());

  DSizes<Index, ReshapedDims> reshaped_dims;
  reshaped_dims[shuffle[0]] = dims[0] * dims[1];
  for (int i = 1; i < ReshapedDims; ++i) reshaped_dims[shuffle[i]] = dims[i + 1];

  Tensor<T, ReshapedDims, Options, Index> golden = src.reshape(reshaped_dims);

  // Now reshape using configured tensor executor.
  Tensor<T, ReshapedDims, Options, Index> dst(golden.dimensions());

  auto expr = src.reshape(reshaped_dims);

  using Assign = TensorAssignOp<decltype(dst), const decltype(expr)>;
  using Executor =
      internal::TensorExecutor<const Assign, Device, Vectorizable, Tiling>;

  Executor::run(Assign(dst, expr), d);

  for (Index i = 0; i < dst.dimensions().TotalSize(); ++i) {
    VERIFY_IS_EQUAL(dst.coeff(i), golden.coeff(i));
  }
}

template <typename T, int NumDims, typename Device, bool Vectorizable,
          TiledEvaluation Tiling, int Layout>
static void test_execute_slice_rvalue(Device d)
{
  static_assert(NumDims >= 2, "NumDims must be greater or equal than 2");
  static constexpr int Options = 0 | Layout;

  auto dims = RandomDims<NumDims>(5, 10);
  Tensor<T, NumDims, Options, Index> src(dims);
  src.setRandom();

  // Pick a random slice of src tensor.
  auto slice_start = DSizes<Index, NumDims>(RandomDims<NumDims>());
  auto slice_size = DSizes<Index, NumDims>(RandomDims<NumDims>());

  // Make sure that slice start + size do not overflow tensor dims.
  for (int i = 0; i < NumDims; ++i) {
    slice_start[i] = numext::mini(dims[i] - 1, slice_start[i]);
    slice_size[i] = numext::mini(slice_size[i], dims[i] - slice_start[i]);
  }

  Tensor<T, NumDims, Options, Index> golden =
      src.slice(slice_start, slice_size);

  // Now reshape using configured tensor executor.
  Tensor<T, NumDims, Options, Index> dst(golden.dimensions());

  auto expr = src.slice(slice_start, slice_size);

  using Assign = TensorAssignOp<decltype(dst), const decltype(expr)>;
  using Executor =
      internal::TensorExecutor<const Assign, Device, Vectorizable, Tiling>;

  Executor::run(Assign(dst, expr), d);

  for (Index i = 0; i < dst.dimensions().TotalSize(); ++i) {
    VERIFY_IS_EQUAL(dst.coeff(i), golden.coeff(i));
  }
}

template <typename T, int NumDims, typename Device, bool Vectorizable,
    TiledEvaluation Tiling, int Layout>
static void test_execute_slice_lvalue(Device d)
{
  static_assert(NumDims >= 2, "NumDims must be greater or equal than 2");
  static constexpr int Options = 0 | Layout;

  auto dims = RandomDims<NumDims>(5, 10);
  Tensor<T, NumDims, Options, Index> src(dims);
  src.setRandom();

  // Pick a random slice of src tensor.
  auto slice_start = DSizes<Index, NumDims>(RandomDims<NumDims>(1, 10));
  auto slice_size = DSizes<Index, NumDims>(RandomDims<NumDims>(1, 10));

  // Make sure that slice start + size do not overflow tensor dims.
  for (int i = 0; i < NumDims; ++i) {
    slice_start[i] = numext::mini(dims[i] - 1, slice_start[i]);
    slice_size[i] = numext::mini(slice_size[i], dims[i] - slice_start[i]);
  }

  Tensor<T, NumDims, Options, Index> slice(slice_size);
  slice.setRandom();

  // Assign a slice using default executor.
  Tensor<T, NumDims, Options, Index> golden = src;
  golden.slice(slice_start, slice_size) = slice;

  // And using configured execution strategy.
  Tensor<T, NumDims, Options, Index> dst = src;
  auto expr = dst.slice(slice_start, slice_size);

  using Assign = TensorAssignOp<decltype(expr), const decltype(slice)>;
  using Executor =
      internal::TensorExecutor<const Assign, Device, Vectorizable, Tiling>;

  Executor::run(Assign(expr, slice), d);

  for (Index i = 0; i < dst.dimensions().TotalSize(); ++i) {
    VERIFY_IS_EQUAL(dst.coeff(i), golden.coeff(i));
  }
}

template <typename T, int NumDims, typename Device, bool Vectorizable,
    TiledEvaluation Tiling, int Layout>
static void test_execute_broadcasting_of_forced_eval(Device d)
{
  static constexpr int Options = 0 | Layout;

  auto dims = RandomDims<NumDims>(1, 10);
  Tensor<T, NumDims, Options, Index> src(dims);
  src.setRandom();

  const auto broadcasts = RandomDims<NumDims>(1, 7);
  const auto expr = src.square().eval().broadcast(broadcasts);

  // We assume that broadcasting on a default device is tested and correct, so
  // we can rely on it to verify correctness of tensor executor and tiling.
  Tensor<T, NumDims, Options, Index> golden;
  golden = expr;

  // Now do the broadcasting using configured tensor executor.
  Tensor<T, NumDims, Options, Index> dst(golden.dimensions());

  using Assign = TensorAssignOp<decltype(dst), const decltype(expr)>;
  using Executor =
      internal::TensorExecutor<const Assign, Device, Vectorizable, Tiling>;

  Executor::run(Assign(dst, expr), d);

  for (Index i = 0; i < dst.dimensions().TotalSize(); ++i) {
    VERIFY_IS_EQUAL(dst.coeff(i), golden.coeff(i));
  }
}

template<typename T, int NumDims>
struct DummyGenerator {
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  T operator()(const array <Index, NumDims>& dims) const {
    T result = static_cast<T>(0);
    for (int i = 0; i < NumDims; ++i) {
      result += static_cast<T>((i + 1) * dims[i]);
    }
    return result;
  }
};

template <typename T, int NumDims, typename Device, bool Vectorizable,
    TiledEvaluation Tiling, int Layout>
static void test_execute_generator_op(Device d)
{
  static constexpr int Options = 0 | Layout;

  auto dims = RandomDims<NumDims>(20, 30);
  Tensor<T, NumDims, Options, Index> src(dims);
  src.setRandom();

  const auto expr = src.generate(DummyGenerator<T, NumDims>());

  // We assume that generator on a default device is tested and correct, so
  // we can rely on it to verify correctness of tensor executor and tiling.
  Tensor<T, NumDims, Options, Index> golden;
  golden = expr;

  // Now do the broadcasting using configured tensor executor.
  Tensor<T, NumDims, Options, Index> dst(golden.dimensions());

  using Assign = TensorAssignOp<decltype(dst), const decltype(expr)>;
  using Executor =
    internal::TensorExecutor<const Assign, Device, Vectorizable, Tiling>;

  Executor::run(Assign(dst, expr), d);

  for (Index i = 0; i < dst.dimensions().TotalSize(); ++i) {
    VERIFY_IS_EQUAL(dst.coeff(i), golden.coeff(i));
  }
}

template <typename T, int NumDims, typename Device, bool Vectorizable,
    TiledEvaluation Tiling, int Layout>
static void test_execute_reverse_rvalue(Device d)
{
  static constexpr int Options = 0 | Layout;

  auto dims = RandomDims<NumDims>(1, numext::pow(1000000.0, 1.0 / NumDims));
  Tensor <T, NumDims, Options, Index> src(dims);
  src.setRandom();

  // Reverse half of the dimensions.
  Eigen::array<bool, NumDims> reverse;
  for (int i = 0; i < NumDims; ++i) reverse[i] = internal::random<bool>();

  const auto expr = src.reverse(reverse);

  // We assume that reversing on a default device is tested and correct, so
  // we can rely on it to verify correctness of tensor executor and tiling.
  Tensor <T, NumDims, Options, Index> golden;
  golden = expr;

  // Now do the reversing using configured tensor executor.
  Tensor <T, NumDims, Options, Index> dst(golden.dimensions());

  using Assign = TensorAssignOp<decltype(dst), const decltype(expr)>;
  using Executor =
    internal::TensorExecutor<const Assign, Device, Vectorizable, Tiling>;

  Executor::run(Assign(dst, expr), d);

  for (Index i = 0; i < dst.dimensions().TotalSize(); ++i) {
    VERIFY_IS_EQUAL(dst.coeff(i), golden.coeff(i));
  }
}

template <typename T, int NumDims, typename Device, bool Vectorizable,
          TiledEvaluation Tiling, int Layout>
static void test_async_execute_unary_expr(Device d)
{
  static constexpr int Options = 0 | Layout;

  // Pick a large enough tensor size to bypass small tensor block evaluation
  // optimization.
  auto dims = RandomDims<NumDims>(50 / NumDims, 100 / NumDims);

  Tensor<T, NumDims, Options, Index> src(dims);
  Tensor<T, NumDims, Options, Index> dst(dims);

  src.setRandom();
  const auto expr = src.square();

  Eigen::Barrier done(1);
  auto on_done = [&done]() { done.Notify(); };

  using Assign = TensorAssignOp<decltype(dst), const decltype(expr)>;
  using DoneCallback = decltype(on_done);
  using Executor = internal::TensorAsyncExecutor<const Assign, Device, DoneCallback,
                                                 Vectorizable, Tiling>;

  Executor::runAsync(Assign(dst, expr), d, on_done);
  done.Wait();

  for (Index i = 0; i < dst.dimensions().TotalSize(); ++i) {
    T square = src.coeff(i) * src.coeff(i);
    VERIFY_IS_EQUAL(square, dst.coeff(i));
  }
}

template <typename T, int NumDims, typename Device, bool Vectorizable,
          TiledEvaluation Tiling, int Layout>
static void test_async_execute_binary_expr(Device d)
{
  static constexpr int Options = 0 | Layout;

  // Pick a large enough tensor size to bypass small tensor block evaluation
  // optimization.
  auto dims = RandomDims<NumDims>(50 / NumDims, 100 / NumDims);

  Tensor<T, NumDims, Options, Index> lhs(dims);
  Tensor<T, NumDims, Options, Index> rhs(dims);
  Tensor<T, NumDims, Options, Index> dst(dims);

  lhs.setRandom();
  rhs.setRandom();

  const auto expr = lhs + rhs;

  Eigen::Barrier done(1);
  auto on_done = [&done]() { done.Notify(); };

  using Assign = TensorAssignOp<decltype(dst), const decltype(expr)>;
  using DoneCallback = decltype(on_done);
  using Executor = internal::TensorAsyncExecutor<const Assign, Device, DoneCallback,
                                                 Vectorizable, Tiling>;

  Executor::runAsync(Assign(dst, expr), d, on_done);
  done.Wait();

  for (Index i = 0; i < dst.dimensions().TotalSize(); ++i) {
    T sum = lhs.coeff(i) + rhs.coeff(i);
    VERIFY_IS_EQUAL(sum, dst.coeff(i));
  }
}

#ifdef EIGEN_DONT_VECTORIZE
#define VECTORIZABLE(VAL) !EIGEN_DONT_VECTORIZE && VAL
#else
#define VECTORIZABLE(VAL) VAL
#endif

#define CALL_SUBTEST_PART(PART) \
  CALL_SUBTEST_##PART

#define CALL_SUBTEST_COMBINATIONS(PART, NAME, T, NUM_DIMS)                                                                                 \
  CALL_SUBTEST_PART(PART)((NAME<T, NUM_DIMS, DefaultDevice,    false,               TiledEvaluation::Off,     ColMajor>(default_device))); \
  CALL_SUBTEST_PART(PART)((NAME<T, NUM_DIMS, DefaultDevice,    false,               TiledEvaluation::On,  ColMajor>(default_device)));     \
  CALL_SUBTEST_PART(PART)((NAME<T, NUM_DIMS, DefaultDevice,    VECTORIZABLE(true),  TiledEvaluation::Off,     ColMajor>(default_device))); \
  CALL_SUBTEST_PART(PART)((NAME<T, NUM_DIMS, DefaultDevice,    VECTORIZABLE(true),  TiledEvaluation::On,  ColMajor>(default_device)));     \
  CALL_SUBTEST_PART(PART)((NAME<T, NUM_DIMS, DefaultDevice,    false,               TiledEvaluation::Off,     RowMajor>(default_device))); \
  CALL_SUBTEST_PART(PART)((NAME<T, NUM_DIMS, DefaultDevice,    false,               TiledEvaluation::On,  RowMajor>(default_device)));     \
  CALL_SUBTEST_PART(PART)((NAME<T, NUM_DIMS, DefaultDevice,    VECTORIZABLE(true),  TiledEvaluation::Off,     RowMajor>(default_device))); \
  CALL_SUBTEST_PART(PART)((NAME<T, NUM_DIMS, DefaultDevice,    VECTORIZABLE(true),  TiledEvaluation::On,  RowMajor>(default_device)));     \
  CALL_SUBTEST_PART(PART)((NAME<T, NUM_DIMS, ThreadPoolDevice, false,               TiledEvaluation::Off,     ColMajor>(tp_device)));      \
  CALL_SUBTEST_PART(PART)((NAME<T, NUM_DIMS, ThreadPoolDevice, false,               TiledEvaluation::On,  ColMajor>(tp_device)));          \
  CALL_SUBTEST_PART(PART)((NAME<T, NUM_DIMS, ThreadPoolDevice, VECTORIZABLE(true),  TiledEvaluation::Off,     ColMajor>(tp_device)));      \
  CALL_SUBTEST_PART(PART)((NAME<T, NUM_DIMS, ThreadPoolDevice, VECTORIZABLE(true),  TiledEvaluation::On,  ColMajor>(tp_device)));          \
  CALL_SUBTEST_PART(PART)((NAME<T, NUM_DIMS, ThreadPoolDevice, false,               TiledEvaluation::Off,     RowMajor>(tp_device)));      \
  CALL_SUBTEST_PART(PART)((NAME<T, NUM_DIMS, ThreadPoolDevice, false,               TiledEvaluation::On,  RowMajor>(tp_device)));          \
  CALL_SUBTEST_PART(PART)((NAME<T, NUM_DIMS, ThreadPoolDevice, VECTORIZABLE(true),  TiledEvaluation::Off,     RowMajor>(tp_device)));      \
  CALL_SUBTEST_PART(PART)((NAME<T, NUM_DIMS, ThreadPoolDevice, VECTORIZABLE(true),  TiledEvaluation::On,  RowMajor>(tp_device)))

// NOTE: Currently only ThreadPoolDevice supports async expression evaluation.
#define CALL_ASYNC_SUBTEST_COMBINATIONS(PART, NAME, T, NUM_DIMS)                                                                      \
  CALL_SUBTEST_PART(PART)((NAME<T, NUM_DIMS, ThreadPoolDevice, false,               TiledEvaluation::Off,     ColMajor>(tp_device))); \
  CALL_SUBTEST_PART(PART)((NAME<T, NUM_DIMS, ThreadPoolDevice, false,               TiledEvaluation::On,  ColMajor>(tp_device)));     \
  CALL_SUBTEST_PART(PART)((NAME<T, NUM_DIMS, ThreadPoolDevice, VECTORIZABLE(true),  TiledEvaluation::Off,     ColMajor>(tp_device))); \
  CALL_SUBTEST_PART(PART)((NAME<T, NUM_DIMS, ThreadPoolDevice, VECTORIZABLE(true),  TiledEvaluation::On,  ColMajor>(tp_device)));     \
  CALL_SUBTEST_PART(PART)((NAME<T, NUM_DIMS, ThreadPoolDevice, false,               TiledEvaluation::Off,     RowMajor>(tp_device))); \
  CALL_SUBTEST_PART(PART)((NAME<T, NUM_DIMS, ThreadPoolDevice, false,               TiledEvaluation::On,  RowMajor>(tp_device)));     \
  CALL_SUBTEST_PART(PART)((NAME<T, NUM_DIMS, ThreadPoolDevice, VECTORIZABLE(true),  TiledEvaluation::Off,     RowMajor>(tp_device))); \
  CALL_SUBTEST_PART(PART)((NAME<T, NUM_DIMS, ThreadPoolDevice, VECTORIZABLE(true),  TiledEvaluation::On,  RowMajor>(tp_device)))

EIGEN_DECLARE_TEST(cxx11_tensor_executor) {
  Eigen::DefaultDevice default_device;
  // Default device is unused in ASYNC tests.
  EIGEN_UNUSED_VARIABLE(default_device);

  const auto num_threads = internal::random<int>(20, 24);
  Eigen::ThreadPool tp(num_threads);
  Eigen::ThreadPoolDevice tp_device(&tp, num_threads);

  CALL_SUBTEST_COMBINATIONS(1, test_execute_unary_expr, float, 3);
  CALL_SUBTEST_COMBINATIONS(1, test_execute_unary_expr, float, 4);
  CALL_SUBTEST_COMBINATIONS(1, test_execute_unary_expr, float, 5);

  CALL_SUBTEST_COMBINATIONS(2, test_execute_binary_expr, float, 3);
  CALL_SUBTEST_COMBINATIONS(2, test_execute_binary_expr, float, 4);
  CALL_SUBTEST_COMBINATIONS(2, test_execute_binary_expr, float, 5);

  CALL_SUBTEST_COMBINATIONS(3, test_execute_broadcasting, float, 3);
  CALL_SUBTEST_COMBINATIONS(3, test_execute_broadcasting, float, 4);
  CALL_SUBTEST_COMBINATIONS(3, test_execute_broadcasting, float, 5);

  CALL_SUBTEST_COMBINATIONS(4, test_execute_chipping_rvalue, float, 3);
  CALL_SUBTEST_COMBINATIONS(4, test_execute_chipping_rvalue, float, 4);
  CALL_SUBTEST_COMBINATIONS(4, test_execute_chipping_rvalue, float, 5);

  CALL_SUBTEST_COMBINATIONS(5, test_execute_chipping_lvalue, float, 3);
  CALL_SUBTEST_COMBINATIONS(5, test_execute_chipping_lvalue, float, 4);
  CALL_SUBTEST_COMBINATIONS(5, test_execute_chipping_lvalue, float, 5);

  CALL_SUBTEST_COMBINATIONS(6, test_execute_shuffle_rvalue, float, 3);
  CALL_SUBTEST_COMBINATIONS(6, test_execute_shuffle_rvalue, float, 4);
  CALL_SUBTEST_COMBINATIONS(6, test_execute_shuffle_rvalue, float, 5);

  CALL_SUBTEST_COMBINATIONS(7, test_execute_shuffle_lvalue, float, 3);
  CALL_SUBTEST_COMBINATIONS(7, test_execute_shuffle_lvalue, float, 4);
  CALL_SUBTEST_COMBINATIONS(7, test_execute_shuffle_lvalue, float, 5);

  CALL_SUBTEST_COMBINATIONS(9, test_execute_reshape, float, 2);
  CALL_SUBTEST_COMBINATIONS(9, test_execute_reshape, float, 3);
  CALL_SUBTEST_COMBINATIONS(9, test_execute_reshape, float, 4);
  CALL_SUBTEST_COMBINATIONS(9, test_execute_reshape, float, 5);

  CALL_SUBTEST_COMBINATIONS(10, test_execute_slice_rvalue, float, 2);
  CALL_SUBTEST_COMBINATIONS(10, test_execute_slice_rvalue, float, 3);
  CALL_SUBTEST_COMBINATIONS(10, test_execute_slice_rvalue, float, 4);
  CALL_SUBTEST_COMBINATIONS(10, test_execute_slice_rvalue, float, 5);

  CALL_SUBTEST_COMBINATIONS(11, test_execute_slice_lvalue, float, 2);
  CALL_SUBTEST_COMBINATIONS(11, test_execute_slice_lvalue, float, 3);
  CALL_SUBTEST_COMBINATIONS(11, test_execute_slice_lvalue, float, 4);
  CALL_SUBTEST_COMBINATIONS(11, test_execute_slice_lvalue, float, 5);

  CALL_SUBTEST_COMBINATIONS(12, test_execute_broadcasting_of_forced_eval, float, 2);
  CALL_SUBTEST_COMBINATIONS(12, test_execute_broadcasting_of_forced_eval, float, 3);
  CALL_SUBTEST_COMBINATIONS(12, test_execute_broadcasting_of_forced_eval, float, 4);
  CALL_SUBTEST_COMBINATIONS(12, test_execute_broadcasting_of_forced_eval, float, 5);

  CALL_SUBTEST_COMBINATIONS(13, test_execute_generator_op, float, 2);
  CALL_SUBTEST_COMBINATIONS(13, test_execute_generator_op, float, 3);
  CALL_SUBTEST_COMBINATIONS(13, test_execute_generator_op, float, 4);
  CALL_SUBTEST_COMBINATIONS(13, test_execute_generator_op, float, 5);

  CALL_SUBTEST_COMBINATIONS(14, test_execute_reverse_rvalue, float, 1);
  CALL_SUBTEST_COMBINATIONS(14, test_execute_reverse_rvalue, float, 2);
  CALL_SUBTEST_COMBINATIONS(14, test_execute_reverse_rvalue, float, 3);
  CALL_SUBTEST_COMBINATIONS(14, test_execute_reverse_rvalue, float, 4);
  CALL_SUBTEST_COMBINATIONS(14, test_execute_reverse_rvalue, float, 5);

  CALL_ASYNC_SUBTEST_COMBINATIONS(15, test_async_execute_unary_expr, float, 3);
  CALL_ASYNC_SUBTEST_COMBINATIONS(15, test_async_execute_unary_expr, float, 4);
  CALL_ASYNC_SUBTEST_COMBINATIONS(15, test_async_execute_unary_expr, float, 5);

  CALL_ASYNC_SUBTEST_COMBINATIONS(16, test_async_execute_binary_expr, float, 3);
  CALL_ASYNC_SUBTEST_COMBINATIONS(16, test_async_execute_binary_expr, float, 4);
  CALL_ASYNC_SUBTEST_COMBINATIONS(16, test_async_execute_binary_expr, float, 5);

  // Force CMake to split this test.
  // EIGEN_SUFFIXES;1;2;3;4;5;6;7;8;9;10;11;12;13;14;15;16
}
