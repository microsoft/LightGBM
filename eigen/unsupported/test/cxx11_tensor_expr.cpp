// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <numeric>

#include "main.h"

#include <Eigen/CXX11/Tensor>

using Eigen::Tensor;
using Eigen::RowMajor;

static void test_1d()
{
  Tensor<float, 1> vec1(6);
  Tensor<float, 1, RowMajor> vec2(6);

  vec1(0) = 4.0;  vec2(0) = 0.0;
  vec1(1) = 8.0;  vec2(1) = 1.0;
  vec1(2) = 15.0; vec2(2) = 2.0;
  vec1(3) = 16.0; vec2(3) = 3.0;
  vec1(4) = 23.0; vec2(4) = 4.0;
  vec1(5) = 42.0; vec2(5) = 5.0;

  float data3[6];
  TensorMap<Tensor<float, 1>> vec3(data3, 6);
  vec3 = vec1.sqrt();
  float data4[6];
  TensorMap<Tensor<float, 1, RowMajor>> vec4(data4, 6);
  vec4 = vec2.square();
  float data5[6];
  TensorMap<Tensor<float, 1, RowMajor>> vec5(data5, 6);
  vec5 = vec2.cube();

  VERIFY_IS_APPROX(vec3(0), sqrtf(4.0));
  VERIFY_IS_APPROX(vec3(1), sqrtf(8.0));
  VERIFY_IS_APPROX(vec3(2), sqrtf(15.0));
  VERIFY_IS_APPROX(vec3(3), sqrtf(16.0));
  VERIFY_IS_APPROX(vec3(4), sqrtf(23.0));
  VERIFY_IS_APPROX(vec3(5), sqrtf(42.0));

  VERIFY_IS_APPROX(vec4(0), 0.0f);
  VERIFY_IS_APPROX(vec4(1), 1.0f);
  VERIFY_IS_APPROX(vec4(2), 2.0f * 2.0f);
  VERIFY_IS_APPROX(vec4(3), 3.0f * 3.0f);
  VERIFY_IS_APPROX(vec4(4), 4.0f * 4.0f);
  VERIFY_IS_APPROX(vec4(5), 5.0f * 5.0f);

  VERIFY_IS_APPROX(vec5(0), 0.0f);
  VERIFY_IS_APPROX(vec5(1), 1.0f);
  VERIFY_IS_APPROX(vec5(2), 2.0f * 2.0f * 2.0f);
  VERIFY_IS_APPROX(vec5(3), 3.0f * 3.0f * 3.0f);
  VERIFY_IS_APPROX(vec5(4), 4.0f * 4.0f * 4.0f);
  VERIFY_IS_APPROX(vec5(5), 5.0f * 5.0f * 5.0f);

  vec3 = vec1 + vec2;
  VERIFY_IS_APPROX(vec3(0), 4.0f + 0.0f);
  VERIFY_IS_APPROX(vec3(1), 8.0f + 1.0f);
  VERIFY_IS_APPROX(vec3(2), 15.0f + 2.0f);
  VERIFY_IS_APPROX(vec3(3), 16.0f + 3.0f);
  VERIFY_IS_APPROX(vec3(4), 23.0f + 4.0f);
  VERIFY_IS_APPROX(vec3(5), 42.0f + 5.0f);
}

static void test_2d()
{
  float data1[6];
  TensorMap<Tensor<float, 2>> mat1(data1, 2, 3);
  float data2[6];
  TensorMap<Tensor<float, 2, RowMajor>> mat2(data2, 2, 3);

  mat1(0,0) = 0.0;
  mat1(0,1) = 1.0;
  mat1(0,2) = 2.0;
  mat1(1,0) = 3.0;
  mat1(1,1) = 4.0;
  mat1(1,2) = 5.0;

  mat2(0,0) = -0.0;
  mat2(0,1) = -1.0;
  mat2(0,2) = -2.0;
  mat2(1,0) = -3.0;
  mat2(1,1) = -4.0;
  mat2(1,2) = -5.0;

  Tensor<float, 2> mat3(2,3);
  Tensor<float, 2, RowMajor> mat4(2,3);
  mat3 = mat1.abs();
  mat4 = mat2.abs();

  VERIFY_IS_APPROX(mat3(0,0), 0.0f);
  VERIFY_IS_APPROX(mat3(0,1), 1.0f);
  VERIFY_IS_APPROX(mat3(0,2), 2.0f);
  VERIFY_IS_APPROX(mat3(1,0), 3.0f);
  VERIFY_IS_APPROX(mat3(1,1), 4.0f);
  VERIFY_IS_APPROX(mat3(1,2), 5.0f);

  VERIFY_IS_APPROX(mat4(0,0), 0.0f);
  VERIFY_IS_APPROX(mat4(0,1), 1.0f);
  VERIFY_IS_APPROX(mat4(0,2), 2.0f);
  VERIFY_IS_APPROX(mat4(1,0), 3.0f);
  VERIFY_IS_APPROX(mat4(1,1), 4.0f);
  VERIFY_IS_APPROX(mat4(1,2), 5.0f);
}

static void test_3d()
{
  Tensor<float, 3> mat1(2,3,7);
  Tensor<float, 3, RowMajor> mat2(2,3,7);

  float val = 1.0f;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 7; ++k) {
        mat1(i,j,k) = val;
        mat2(i,j,k) = val;
        val += 1.0f;
      }
    }
  }

  Tensor<float, 3> mat3(2,3,7);
  mat3 = mat1 + mat1;
  Tensor<float, 3, RowMajor> mat4(2,3,7);
  mat4 = mat2 * 3.14f;
  Tensor<float, 3> mat5(2,3,7);
  mat5 = mat1.inverse().log();
  Tensor<float, 3, RowMajor> mat6(2,3,7);
  mat6 = mat2.pow(0.5f) * 3.14f;
  Tensor<float, 3> mat7(2,3,7);
  mat7 = mat1.cwiseMax(mat5 * 2.0f).exp();
  Tensor<float, 3, RowMajor> mat8(2,3,7);
  mat8 = (-mat2).exp() * 3.14f;
  Tensor<float, 3, RowMajor> mat9(2,3,7);
  mat9 = mat2 + 3.14f;
  Tensor<float, 3, RowMajor> mat10(2,3,7);
  mat10 = mat2 - 3.14f;
  Tensor<float, 3, RowMajor> mat11(2,3,7);
  mat11 = mat2 / 3.14f;

  val = 1.0f;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 7; ++k) {
        VERIFY_IS_APPROX(mat3(i,j,k), val + val);
        VERIFY_IS_APPROX(mat4(i,j,k), val * 3.14f);
        VERIFY_IS_APPROX(mat5(i,j,k), logf(1.0f/val));
        VERIFY_IS_APPROX(mat6(i,j,k), sqrtf(val) * 3.14f);
        VERIFY_IS_APPROX(mat7(i,j,k), expf((std::max)(val, mat5(i,j,k) * 2.0f)));
        VERIFY_IS_APPROX(mat8(i,j,k), expf(-val) * 3.14f);
        VERIFY_IS_APPROX(mat9(i,j,k), val + 3.14f);
        VERIFY_IS_APPROX(mat10(i,j,k), val - 3.14f);
        VERIFY_IS_APPROX(mat11(i,j,k), val / 3.14f);
        val += 1.0f;
      }
    }
  }
}

static void test_constants()
{
  Tensor<float, 3> mat1(2,3,7);
  Tensor<float, 3> mat2(2,3,7);
  Tensor<float, 3> mat3(2,3,7);

  float val = 1.0f;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 7; ++k) {
        mat1(i,j,k) = val;
        val += 1.0f;
      }
    }
  }
  mat2 = mat1.constant(3.14f);
  mat3 = mat1.cwiseMax(7.3f).exp();

  val = 1.0f;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 7; ++k) {
        VERIFY_IS_APPROX(mat2(i,j,k), 3.14f);
        VERIFY_IS_APPROX(mat3(i,j,k), expf((std::max)(val, 7.3f)));
        val += 1.0f;
      }
    }
  }
}

static void test_boolean()
{
  const int kSize = 31;
  Tensor<int, 1> vec(kSize);
  std::iota(vec.data(), vec.data() + kSize, 0);

  // Test ||.
  Tensor<bool, 1> bool1 = vec < vec.constant(1) || vec > vec.constant(4);
  for (int i = 0; i < kSize; ++i) {
    bool expected = i < 1 || i > 4;
    VERIFY_IS_EQUAL(bool1[i], expected);
  }

  // Test &&, including cast of operand vec.
  Tensor<bool, 1> bool2 = vec.cast<bool>() && vec < vec.constant(4);
  for (int i = 0; i < kSize; ++i) {
    bool expected = bool(i) && i < 4;
    VERIFY_IS_EQUAL(bool2[i], expected);
  }

  // Compilation tests:
  // Test Tensor<bool> against results of cast or comparison; verifies that
  // CoeffReturnType is set to match Op return type of bool for Unary and Binary
  // Ops.
  Tensor<bool, 1> bool3 = vec.cast<bool>() && bool2;
  bool3 = vec < vec.constant(4) && bool2;
}

static void test_functors()
{
  Tensor<float, 3> mat1(2,3,7);
  Tensor<float, 3> mat2(2,3,7);
  Tensor<float, 3> mat3(2,3,7);

  float val = 1.0f;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 7; ++k) {
        mat1(i,j,k) = val;
        val += 1.0f;
      }
    }
  }
  mat2 = mat1.inverse().unaryExpr(&asinf);
  mat3 = mat1.unaryExpr(&tanhf);

  val = 1.0f;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 7; ++k) {
        VERIFY_IS_APPROX(mat2(i,j,k), asinf(1.0f / mat1(i,j,k)));
        VERIFY_IS_APPROX(mat3(i,j,k), tanhf(mat1(i,j,k)));
        val += 1.0f;
      }
    }
  }
}

static void test_type_casting()
{
  Tensor<bool, 3> mat1(2,3,7);
  Tensor<float, 3> mat2(2,3,7);
  Tensor<double, 3> mat3(2,3,7);
  mat1.setRandom();
  mat2.setRandom();

  mat3 = mat1.cast<double>();
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 7; ++k) {
        VERIFY_IS_APPROX(mat3(i,j,k), mat1(i,j,k) ? 1.0 : 0.0);
      }
    }
  }

  mat3 = mat2.cast<double>();
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 7; ++k) {
        VERIFY_IS_APPROX(mat3(i,j,k), static_cast<double>(mat2(i,j,k)));
      }
    }
  }
}

static void test_select()
{
  Tensor<float, 3> selector(2,3,7);
  Tensor<float, 3> mat1(2,3,7);
  Tensor<float, 3> mat2(2,3,7);
  Tensor<float, 3> result(2,3,7);

  selector.setRandom();
  mat1.setRandom();
  mat2.setRandom();
  result = (selector > selector.constant(0.5f)).select(mat1, mat2);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 7; ++k) {
        VERIFY_IS_APPROX(result(i,j,k), (selector(i,j,k) > 0.5f) ? mat1(i,j,k) : mat2(i,j,k));
      }
    }
  }
}

template <typename Scalar>
void test_minmax_nan_propagation_templ() {
  for (int size = 1; size < 17; ++size) {
    const Scalar kNan = std::numeric_limits<Scalar>::quiet_NaN();
    Tensor<Scalar, 1> vec_nan(size);
    Tensor<Scalar, 1> vec_zero(size);
    Tensor<Scalar, 1> vec_res(size);
    vec_nan.setConstant(kNan);
    vec_zero.setZero();
    vec_res.setZero();

    // Test that we propagate NaNs in the tensor when applying the
    // cwiseMax(scalar) operator, which is used for the Relu operator.
    vec_res = vec_nan.cwiseMax(Scalar(0));
    for (int i = 0; i < size; ++i) {
      VERIFY((numext::isnan)(vec_res(i)));
    }

    // Test that NaNs do not propagate if we reverse the arguments.
    vec_res = vec_zero.cwiseMax(kNan);
    for (int i = 0; i < size; ++i) {
      VERIFY_IS_EQUAL(vec_res(i), Scalar(0));
    }

    // Test that we propagate NaNs in the tensor when applying the
    // cwiseMin(scalar) operator.
    vec_res.setZero();
    vec_res = vec_nan.cwiseMin(Scalar(0));
    for (int i = 0; i < size; ++i) {
      VERIFY((numext::isnan)(vec_res(i)));
    }

    // Test that NaNs do not propagate if we reverse the arguments.
    vec_res = vec_zero.cwiseMin(kNan);
    for (int i = 0; i < size; ++i) {
      VERIFY_IS_EQUAL(vec_res(i), Scalar(0));
    }
  }
}

static void test_clip()
{
  Tensor<float, 1> vec(6);
  vec(0) = 4.0;
  vec(1) = 8.0;
  vec(2) = 15.0;
  vec(3) = 16.0;
  vec(4) = 23.0;
  vec(5) = 42.0;

  float kMin = 20;
  float kMax = 30;

  Tensor<float, 1> vec_clipped(6);
  vec_clipped = vec.clip(kMin, kMax);
  for (int i = 0; i < 6; ++i) {
    VERIFY_IS_EQUAL(vec_clipped(i), numext::mini(numext::maxi(vec(i), kMin), kMax));
  }
}

static void test_minmax_nan_propagation()
{
  test_minmax_nan_propagation_templ<float>();
  test_minmax_nan_propagation_templ<double>();
}

EIGEN_DECLARE_TEST(cxx11_tensor_expr)
{
  CALL_SUBTEST(test_1d());
  CALL_SUBTEST(test_2d());
  CALL_SUBTEST(test_3d());
  CALL_SUBTEST(test_constants());
  CALL_SUBTEST(test_boolean());
  CALL_SUBTEST(test_functors());
  CALL_SUBTEST(test_type_casting());
  CALL_SUBTEST(test_select());
  CALL_SUBTEST(test_clip());
  CALL_SUBTEST(test_minmax_nan_propagation());
}
