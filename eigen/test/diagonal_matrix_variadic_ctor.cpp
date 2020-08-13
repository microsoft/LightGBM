// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2019 David Tellenbach <david.tellenbach@tellnotes.org>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_NO_STATIC_ASSERT

#include "main.h"

template <typename Scalar>
void assertionTest()
{
  typedef DiagonalMatrix<Scalar, 5> DiagMatrix5;
  typedef DiagonalMatrix<Scalar, 7> DiagMatrix7;
  typedef DiagonalMatrix<Scalar, Dynamic> DiagMatrixX;

  Scalar raw[6];
  for (int i = 0; i < 6; ++i) {
    raw[i] = internal::random<Scalar>();
  }

  VERIFY_RAISES_ASSERT((DiagMatrix5{raw[0], raw[1], raw[2], raw[3]}));
  VERIFY_RAISES_ASSERT((DiagMatrix5{raw[0], raw[1], raw[3]}));
  VERIFY_RAISES_ASSERT((DiagMatrix7{raw[0], raw[1], raw[2], raw[3]}));

  VERIFY_RAISES_ASSERT((DiagMatrixX {
    {raw[0], raw[1], raw[2]},
    {raw[3], raw[4], raw[5]}
  }));
}

#define VERIFY_IMPLICIT_CONVERSION_3(DIAGTYPE, V0, V1, V2) \
  DIAGTYPE d(V0, V1, V2);                                  \
  DIAGTYPE::DenseMatrixType Dense = d.toDenseMatrix();     \
  VERIFY_IS_APPROX(Dense(0, 0), (Scalar)V0);               \
  VERIFY_IS_APPROX(Dense(1, 1), (Scalar)V1);               \
  VERIFY_IS_APPROX(Dense(2, 2), (Scalar)V2);

#define VERIFY_IMPLICIT_CONVERSION_4(DIAGTYPE, V0, V1, V2, V3) \
  DIAGTYPE d(V0, V1, V2, V3);                                  \
  DIAGTYPE::DenseMatrixType Dense = d.toDenseMatrix();         \
  VERIFY_IS_APPROX(Dense(0, 0), (Scalar)V0);                   \
  VERIFY_IS_APPROX(Dense(1, 1), (Scalar)V1);                   \
  VERIFY_IS_APPROX(Dense(2, 2), (Scalar)V2);                   \
  VERIFY_IS_APPROX(Dense(3, 3), (Scalar)V3);

#define VERIFY_IMPLICIT_CONVERSION_5(DIAGTYPE, V0, V1, V2, V3, V4) \
  DIAGTYPE d(V0, V1, V2, V3, V4);                                  \
  DIAGTYPE::DenseMatrixType Dense = d.toDenseMatrix();             \
  VERIFY_IS_APPROX(Dense(0, 0), (Scalar)V0);                       \
  VERIFY_IS_APPROX(Dense(1, 1), (Scalar)V1);                       \
  VERIFY_IS_APPROX(Dense(2, 2), (Scalar)V2);                       \
  VERIFY_IS_APPROX(Dense(3, 3), (Scalar)V3);                       \
  VERIFY_IS_APPROX(Dense(4, 4), (Scalar)V4);

template<typename Scalar>
void constructorTest()
{
  typedef DiagonalMatrix<Scalar, 0> DiagonalMatrix0;
  typedef DiagonalMatrix<Scalar, 3> DiagonalMatrix3;
  typedef DiagonalMatrix<Scalar, 4> DiagonalMatrix4;
  typedef DiagonalMatrix<Scalar, Dynamic> DiagonalMatrixX;

  Scalar raw[7];
  for (int k = 0; k < 7; ++k) raw[k] = internal::random<Scalar>();

  // Fixed-sized matrices
  {
    DiagonalMatrix0 a {{}};
    VERIFY(a.rows() == 0);
    VERIFY(a.cols() == 0);
    typename DiagonalMatrix0::DenseMatrixType m = a.toDenseMatrix();
    for (Index k = 0; k < a.rows(); ++k) VERIFY(m(k, k) == raw[k]);
  }
  {
    DiagonalMatrix3 a {{raw[0], raw[1], raw[2]}};
    VERIFY(a.rows() == 3);
    VERIFY(a.cols() == 3);
    typename DiagonalMatrix3::DenseMatrixType m = a.toDenseMatrix();
    for (Index k = 0; k < a.rows(); ++k) VERIFY(m(k, k) == raw[k]);
  }
  {
    DiagonalMatrix4 a {{raw[0], raw[1], raw[2], raw[3]}};
    VERIFY(a.rows() == 4);
    VERIFY(a.cols() == 4);
    typename DiagonalMatrix4::DenseMatrixType m = a.toDenseMatrix();
    for (Index k = 0; k < a.rows(); ++k) VERIFY(m(k, k) == raw[k]);
  }

  // dynamically sized matrices
  {
    DiagonalMatrixX a{{}};
    VERIFY(a.rows() == 0);
    VERIFY(a.rows() == 0);
    typename DiagonalMatrixX::DenseMatrixType m = a.toDenseMatrix();
    for (Index k = 0; k < a.rows(); ++k) VERIFY(m(k, k) == raw[k]);
  }
  {
    DiagonalMatrixX a{{raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6]}};
    VERIFY(a.rows() == 7);
    VERIFY(a.rows() == 7);
    typename DiagonalMatrixX::DenseMatrixType m = a.toDenseMatrix();
    for (Index k = 0; k < a.rows(); ++k) VERIFY(m(k, k) == raw[k]);
  }
}

template<>
void constructorTest<float>()
{
  typedef float Scalar;

  typedef DiagonalMatrix<Scalar, 0> DiagonalMatrix0;
  typedef DiagonalMatrix<Scalar, 3> DiagonalMatrix3;
  typedef DiagonalMatrix<Scalar, 4> DiagonalMatrix4;
  typedef DiagonalMatrix<Scalar, 5> DiagonalMatrix5;
  typedef DiagonalMatrix<Scalar, Dynamic> DiagonalMatrixX;

  Scalar raw[7];
  for (int k = 0; k < 7; ++k) raw[k] = internal::random<Scalar>();

  // Fixed-sized matrices
  {
    DiagonalMatrix0 a {{}};
    VERIFY(a.rows() == 0);
    VERIFY(a.cols() == 0);
    typename DiagonalMatrix0::DenseMatrixType m = a.toDenseMatrix();
    for (Index k = 0; k < a.rows(); ++k) VERIFY(m(k, k) == raw[k]);
  }
  {
    DiagonalMatrix3 a {{raw[0], raw[1], raw[2]}};
    VERIFY(a.rows() == 3);
    VERIFY(a.cols() == 3);
    typename DiagonalMatrix3::DenseMatrixType m = a.toDenseMatrix();
    for (Index k = 0; k < a.rows(); ++k) VERIFY(m(k, k) == raw[k]);
  }
  {
    DiagonalMatrix4 a {{raw[0], raw[1], raw[2], raw[3]}};
    VERIFY(a.rows() == 4);
    VERIFY(a.cols() == 4);
    typename DiagonalMatrix4::DenseMatrixType m = a.toDenseMatrix();
    for (Index k = 0; k < a.rows(); ++k) VERIFY(m(k, k) == raw[k]);
  }

  // dynamically sized matrices
  {
    DiagonalMatrixX a{{}};
    VERIFY(a.rows() == 0);
    VERIFY(a.rows() == 0);
    typename DiagonalMatrixX::DenseMatrixType m = a.toDenseMatrix();
    for (Index k = 0; k < a.rows(); ++k) VERIFY(m(k, k) == raw[k]);
  }
  {
    DiagonalMatrixX a{{raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6]}};
    VERIFY(a.rows() == 7);
    VERIFY(a.rows() == 7);
    typename DiagonalMatrixX::DenseMatrixType m = a.toDenseMatrix();
    for (Index k = 0; k < a.rows(); ++k) VERIFY(m(k, k) == raw[k]);
  }
  { VERIFY_IMPLICIT_CONVERSION_3(DiagonalMatrix3, 1.2647, 2.56f, -3); }
  { VERIFY_IMPLICIT_CONVERSION_4(DiagonalMatrix4, 1.2647, 2.56f, -3, 3.23f); }
  { VERIFY_IMPLICIT_CONVERSION_5(DiagonalMatrix5, 1.2647, 2.56f, -3, 3.23f, 2); }
}

EIGEN_DECLARE_TEST(diagonal_matrix_variadic_ctor)
{
  CALL_SUBTEST_1(assertionTest<unsigned char>());
  CALL_SUBTEST_1(assertionTest<float>());
  CALL_SUBTEST_1(assertionTest<Index>());
  CALL_SUBTEST_1(assertionTest<int>());
  CALL_SUBTEST_1(assertionTest<long int>());
  CALL_SUBTEST_1(assertionTest<std::ptrdiff_t>());
  CALL_SUBTEST_1(assertionTest<std::complex<double>>());

  CALL_SUBTEST_2(constructorTest<unsigned char>());
  CALL_SUBTEST_2(constructorTest<float>());
  CALL_SUBTEST_2(constructorTest<Index>());
  CALL_SUBTEST_2(constructorTest<int>());
  CALL_SUBTEST_2(constructorTest<long int>());
  CALL_SUBTEST_2(constructorTest<std::ptrdiff_t>());
  CALL_SUBTEST_2(constructorTest<std::complex<double>>());
}
