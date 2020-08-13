// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Gagan Goel <gagan.nith@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#include <Eigen/CXX11/Tensor>

using Eigen::Tensor;
using Eigen::array;

template <int DataLayout>
static void test_0D_trace() {
  Tensor<float, 0, DataLayout> tensor;
  tensor.setRandom();
  array<ptrdiff_t, 0> dims;
  Tensor<float, 0, DataLayout> result = tensor.trace(dims);
  VERIFY_IS_EQUAL(result(), tensor());
}


template <int DataLayout>
static void test_all_dimensions_trace() {
  Tensor<float, 3, DataLayout> tensor1(5, 5, 5);
  tensor1.setRandom();
  Tensor<float, 0, DataLayout> result1 = tensor1.trace();
  VERIFY_IS_EQUAL(result1.rank(), 0);
  float sum = 0.0f;
  for (int i = 0; i < 5; ++i) {
    sum += tensor1(i, i, i);
  }
  VERIFY_IS_EQUAL(result1(), sum);

  Tensor<float, 5, DataLayout> tensor2(7, 7, 7, 7, 7);
  tensor2.setRandom();
  array<ptrdiff_t, 5> dims = { { 2, 1, 0, 3, 4 } };
  Tensor<float, 0, DataLayout> result2 = tensor2.trace(dims);
  VERIFY_IS_EQUAL(result2.rank(), 0);
  sum = 0.0f;
  for (int i = 0; i < 7; ++i) {
    sum += tensor2(i, i, i, i, i);
  }
  VERIFY_IS_EQUAL(result2(), sum);
}


template <int DataLayout>
static void test_simple_trace() {
  Tensor<float, 3, DataLayout> tensor1(3, 5, 3);
  tensor1.setRandom();
  array<ptrdiff_t, 2> dims1 = { { 0, 2 } };
  Tensor<float, 1, DataLayout> result1 = tensor1.trace(dims1);
  VERIFY_IS_EQUAL(result1.rank(), 1);
  VERIFY_IS_EQUAL(result1.dimension(0), 5);
  float sum = 0.0f;
  for (int i = 0; i < 5; ++i) {
    sum = 0.0f;
    for (int j = 0; j < 3; ++j) {
      sum += tensor1(j, i, j);
    }
    VERIFY_IS_EQUAL(result1(i), sum);
  }

  Tensor<float, 4, DataLayout> tensor2(5, 5, 7, 7);
  tensor2.setRandom();
  array<ptrdiff_t, 2> dims2 = { { 2, 3 } };
  Tensor<float, 2, DataLayout> result2 = tensor2.trace(dims2);
  VERIFY_IS_EQUAL(result2.rank(), 2);
  VERIFY_IS_EQUAL(result2.dimension(0), 5);
  VERIFY_IS_EQUAL(result2.dimension(1), 5);
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      sum = 0.0f;
      for (int k = 0; k < 7; ++k) {
        sum += tensor2(i, j, k, k);
      }
      VERIFY_IS_EQUAL(result2(i, j), sum);
    }
  }

  array<ptrdiff_t, 2> dims3 = { { 1, 0 } };
  Tensor<float, 2, DataLayout> result3 = tensor2.trace(dims3);
  VERIFY_IS_EQUAL(result3.rank(), 2);
  VERIFY_IS_EQUAL(result3.dimension(0), 7);
  VERIFY_IS_EQUAL(result3.dimension(1), 7);
  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < 7; ++j) {
      sum = 0.0f;
      for (int k = 0; k < 5; ++k) {
        sum += tensor2(k, k, i, j);
      }
      VERIFY_IS_EQUAL(result3(i, j), sum);
    }
  }

  Tensor<float, 5, DataLayout> tensor3(3, 7, 3, 7, 3);
  tensor3.setRandom();
  array<ptrdiff_t, 3> dims4 = { { 0, 2, 4 } };
  Tensor<float, 2, DataLayout> result4 = tensor3.trace(dims4);
  VERIFY_IS_EQUAL(result4.rank(), 2);
  VERIFY_IS_EQUAL(result4.dimension(0), 7);
  VERIFY_IS_EQUAL(result4.dimension(1), 7);
  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < 7; ++j) {
      sum = 0.0f;
      for (int k = 0; k < 3; ++k) {
        sum += tensor3(k, i, k, j, k);
      }
      VERIFY_IS_EQUAL(result4(i, j), sum);
    }
  }

  Tensor<float, 5, DataLayout> tensor4(3, 7, 4, 7, 5);
  tensor4.setRandom();
  array<ptrdiff_t, 2> dims5 = { { 1, 3 } };
  Tensor<float, 3, DataLayout> result5 = tensor4.trace(dims5);
  VERIFY_IS_EQUAL(result5.rank(), 3);
  VERIFY_IS_EQUAL(result5.dimension(0), 3);
  VERIFY_IS_EQUAL(result5.dimension(1), 4);
  VERIFY_IS_EQUAL(result5.dimension(2), 5);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 5; ++k) {
        sum = 0.0f;
        for (int l = 0; l < 7; ++l) {
          sum += tensor4(i, l, j, l, k);
        }
        VERIFY_IS_EQUAL(result5(i, j, k), sum);
      }
    }
  }
}


template<int DataLayout>
static void test_trace_in_expr() {
  Tensor<float, 4, DataLayout> tensor(2, 3, 5, 3);
  tensor.setRandom();
  array<ptrdiff_t, 2> dims = { { 1, 3 } };
  Tensor<float, 2, DataLayout> result(2, 5);
  result = result.constant(1.0f) - tensor.trace(dims);
  VERIFY_IS_EQUAL(result.rank(), 2);
  VERIFY_IS_EQUAL(result.dimension(0), 2);
  VERIFY_IS_EQUAL(result.dimension(1), 5);
  float sum = 0.0f;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 5; ++j) {
      sum = 0.0f;
      for (int k = 0; k < 3; ++k) {
        sum += tensor(i, k, j, k);
      }
      VERIFY_IS_EQUAL(result(i, j), 1.0f - sum);
    }
  }
}


EIGEN_DECLARE_TEST(cxx11_tensor_trace) {
  CALL_SUBTEST(test_0D_trace<ColMajor>());
  CALL_SUBTEST(test_0D_trace<RowMajor>());
  CALL_SUBTEST(test_all_dimensions_trace<ColMajor>());
  CALL_SUBTEST(test_all_dimensions_trace<RowMajor>());
  CALL_SUBTEST(test_simple_trace<ColMajor>());
  CALL_SUBTEST(test_simple_trace<RowMajor>());
  CALL_SUBTEST(test_trace_in_expr<ColMajor>());
  CALL_SUBTEST(test_trace_in_expr<RowMajor>());
}
