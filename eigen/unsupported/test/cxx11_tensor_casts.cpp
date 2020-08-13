// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include "random_without_cast_overflow.h"

#include <Eigen/CXX11/Tensor>

using Eigen::Tensor;
using Eigen::array;

static void test_simple_cast()
{
  Tensor<float, 2> ftensor(20,30);
  ftensor = ftensor.random() * 100.f;
  Tensor<char, 2> chartensor(20,30);
  chartensor.setRandom();
  Tensor<std::complex<float>, 2> cplextensor(20,30);
  cplextensor.setRandom();

  chartensor = ftensor.cast<char>();
  cplextensor = ftensor.cast<std::complex<float> >();

  for (int i = 0; i < 20; ++i) {
    for (int j = 0; j < 30; ++j) {
      VERIFY_IS_EQUAL(chartensor(i,j), static_cast<char>(ftensor(i,j)));
      VERIFY_IS_EQUAL(cplextensor(i,j), static_cast<std::complex<float> >(ftensor(i,j)));
    }
  }
}


static void test_vectorized_cast()
{
  Tensor<int, 2> itensor(20,30);
  itensor = itensor.random() / 1000;
  Tensor<float, 2> ftensor(20,30);
  ftensor.setRandom();
  Tensor<double, 2> dtensor(20,30);
  dtensor.setRandom();

  ftensor = itensor.cast<float>();
  dtensor = itensor.cast<double>();

  for (int i = 0; i < 20; ++i) {
    for (int j = 0; j < 30; ++j) {
      VERIFY_IS_EQUAL(itensor(i,j), static_cast<int>(ftensor(i,j)));
      VERIFY_IS_EQUAL(dtensor(i,j), static_cast<double>(ftensor(i,j)));
    }
  }
}


static void test_float_to_int_cast()
{
  Tensor<float, 2> ftensor(20,30);
  ftensor = ftensor.random() * 1000.0f;
  Tensor<double, 2> dtensor(20,30);
  dtensor = dtensor.random() * 1000.0;

  Tensor<int, 2> i1tensor = ftensor.cast<int>();
  Tensor<int, 2> i2tensor = dtensor.cast<int>();

  for (int i = 0; i < 20; ++i) {
    for (int j = 0; j < 30; ++j) {
      VERIFY_IS_EQUAL(i1tensor(i,j), static_cast<int>(ftensor(i,j)));
      VERIFY_IS_EQUAL(i2tensor(i,j), static_cast<int>(dtensor(i,j)));
    }
  }
}


static void test_big_to_small_type_cast()
{
  Tensor<double, 2> dtensor(20, 30);
  dtensor.setRandom();
  Tensor<float, 2> ftensor(20, 30);
  ftensor = dtensor.cast<float>();

  for (int i = 0; i < 20; ++i) {
    for (int j = 0; j < 30; ++j) {
      VERIFY_IS_APPROX(dtensor(i,j), static_cast<double>(ftensor(i,j)));
    }
  }
}


static void test_small_to_big_type_cast()
{
  Tensor<float, 2> ftensor(20, 30);
  ftensor.setRandom();
  Tensor<double, 2> dtensor(20, 30);
  dtensor = ftensor.cast<double>();

  for (int i = 0; i < 20; ++i) {
    for (int j = 0; j < 30; ++j) {
      VERIFY_IS_APPROX(dtensor(i,j), static_cast<double>(ftensor(i,j)));
    }
  }
}

template <typename FromType, typename ToType>
static void test_type_cast() {
  Tensor<FromType, 2> ftensor(100, 200);
  // Generate random values for a valid cast.
  for (int i = 0; i < 100; ++i) {
    for (int j = 0; j < 200; ++j) {
      ftensor(i, j) = internal::random_without_cast_overflow<FromType,ToType>::value();
    }
  }

  Tensor<ToType, 2> ttensor(100, 200);
  ttensor = ftensor.template cast<ToType>();

  for (int i = 0; i < 100; ++i) {
    for (int j = 0; j < 200; ++j) {
      const ToType ref = internal::cast<FromType,ToType>(ftensor(i, j));
      VERIFY_IS_APPROX(ttensor(i, j), ref);
    }
  }
}

template<typename Scalar, typename EnableIf = void>
struct test_cast_runner {
  static void run() {
    test_type_cast<Scalar, bool>();
    test_type_cast<Scalar, int8_t>();
    test_type_cast<Scalar, int16_t>();
    test_type_cast<Scalar, int32_t>();
    test_type_cast<Scalar, int64_t>();
    test_type_cast<Scalar, uint8_t>();
    test_type_cast<Scalar, uint16_t>();
    test_type_cast<Scalar, uint32_t>();
    test_type_cast<Scalar, uint64_t>();
    test_type_cast<Scalar, half>();
    test_type_cast<Scalar, bfloat16>();
    test_type_cast<Scalar, float>();
    test_type_cast<Scalar, double>();
    test_type_cast<Scalar, std::complex<float>>();
    test_type_cast<Scalar, std::complex<double>>();
  }
};

// Only certain types allow cast from std::complex<>.
template<typename Scalar>
struct test_cast_runner<Scalar, typename internal::enable_if<NumTraits<Scalar>::IsComplex>::type> {
  static void run() {
    test_type_cast<Scalar, half>();
    test_type_cast<Scalar, bfloat16>();
    test_type_cast<Scalar, std::complex<float>>();
    test_type_cast<Scalar, std::complex<double>>();
  }
};


EIGEN_DECLARE_TEST(cxx11_tensor_casts)
{
  CALL_SUBTEST(test_simple_cast());
  CALL_SUBTEST(test_vectorized_cast());
  CALL_SUBTEST(test_float_to_int_cast());
  CALL_SUBTEST(test_big_to_small_type_cast());
  CALL_SUBTEST(test_small_to_big_type_cast());

  CALL_SUBTEST(test_cast_runner<bool>::run());
  CALL_SUBTEST(test_cast_runner<int8_t>::run());
  CALL_SUBTEST(test_cast_runner<int16_t>::run());
  CALL_SUBTEST(test_cast_runner<int32_t>::run());
  CALL_SUBTEST(test_cast_runner<int64_t>::run());
  CALL_SUBTEST(test_cast_runner<uint8_t>::run());
  CALL_SUBTEST(test_cast_runner<uint16_t>::run());
  CALL_SUBTEST(test_cast_runner<uint32_t>::run());
  CALL_SUBTEST(test_cast_runner<uint64_t>::run());
  CALL_SUBTEST(test_cast_runner<half>::run());
  CALL_SUBTEST(test_cast_runner<bfloat16>::run());
  CALL_SUBTEST(test_cast_runner<float>::run());
  CALL_SUBTEST(test_cast_runner<double>::run());
  CALL_SUBTEST(test_cast_runner<std::complex<float>>::run());
  CALL_SUBTEST(test_cast_runner<std::complex<double>>::run());

}
