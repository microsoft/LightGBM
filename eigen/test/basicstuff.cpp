// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_NO_STATIC_ASSERT

#include "main.h"
#include "random_without_cast_overflow.h"

template<typename MatrixType> void basicStuff(const MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime> SquareMatrixType;

  Index rows = m.rows();
  Index cols = m.cols();

  // this test relies a lot on Random.h, and there's not much more that we can do
  // to test it, hence I consider that we will have tested Random.h
  MatrixType m1 = MatrixType::Random(rows, cols),
             m2 = MatrixType::Random(rows, cols),
             m3(rows, cols),
             mzero = MatrixType::Zero(rows, cols),
             square = Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime>::Random(rows, rows);
  VectorType v1 = VectorType::Random(rows),
             vzero = VectorType::Zero(rows);
  SquareMatrixType sm1 = SquareMatrixType::Random(rows,rows), sm2(rows,rows);

  Scalar x = 0;
  while(x == Scalar(0)) x = internal::random<Scalar>();

  Index r = internal::random<Index>(0, rows-1),
        c = internal::random<Index>(0, cols-1);

  m1.coeffRef(r,c) = x;
  VERIFY_IS_APPROX(x, m1.coeff(r,c));
  m1(r,c) = x;
  VERIFY_IS_APPROX(x, m1(r,c));
  v1.coeffRef(r) = x;
  VERIFY_IS_APPROX(x, v1.coeff(r));
  v1(r) = x;
  VERIFY_IS_APPROX(x, v1(r));
  v1[r] = x;
  VERIFY_IS_APPROX(x, v1[r]);

  // test fetching with various index types.
  Index r1 = internal::random<Index>(0, numext::mini(Index(127),rows-1));
  x = v1(static_cast<char>(r1));
  x = v1(static_cast<signed char>(r1));
  x = v1(static_cast<unsigned char>(r1));
  x = v1(static_cast<signed short>(r1));
  x = v1(static_cast<unsigned short>(r1));
  x = v1(static_cast<signed int>(r1));
  x = v1(static_cast<unsigned int>(r1));
  x = v1(static_cast<signed long>(r1));
  x = v1(static_cast<unsigned long>(r1));
#if EIGEN_HAS_CXX11
  x = v1(static_cast<long long int>(r1));
  x = v1(static_cast<unsigned long long int>(r1));
#endif

  VERIFY_IS_APPROX(               v1,    v1);
  VERIFY_IS_NOT_APPROX(           v1,    2*v1);
  VERIFY_IS_MUCH_SMALLER_THAN(    vzero, v1);
  VERIFY_IS_MUCH_SMALLER_THAN(  vzero, v1.squaredNorm());
  VERIFY_IS_NOT_MUCH_SMALLER_THAN(v1,    v1);
  VERIFY_IS_APPROX(               vzero, v1-v1);
  VERIFY_IS_APPROX(               m1,    m1);
  VERIFY_IS_NOT_APPROX(           m1,    2*m1);
  VERIFY_IS_MUCH_SMALLER_THAN(    mzero, m1);
  VERIFY_IS_NOT_MUCH_SMALLER_THAN(m1,    m1);
  VERIFY_IS_APPROX(               mzero, m1-m1);

  // always test operator() on each read-only expression class,
  // in order to check const-qualifiers.
  // indeed, if an expression class (here Zero) is meant to be read-only,
  // hence has no _write() method, the corresponding MatrixBase method (here zero())
  // should return a const-qualified object so that it is the const-qualified
  // operator() that gets called, which in turn calls _read().
  VERIFY_IS_MUCH_SMALLER_THAN(MatrixType::Zero(rows,cols)(r,c), static_cast<Scalar>(1));

  // now test copying a row-vector into a (column-)vector and conversely.
  square.col(r) = square.row(r).eval();
  Matrix<Scalar, 1, MatrixType::RowsAtCompileTime> rv(rows);
  Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> cv(rows);
  rv = square.row(r);
  cv = square.col(r);

  VERIFY_IS_APPROX(rv, cv.transpose());

  if(cols!=1 && rows!=1 && MatrixType::SizeAtCompileTime!=Dynamic)
  {
    VERIFY_RAISES_ASSERT(m1 = (m2.block(0,0, rows-1, cols-1)));
  }

  if(cols!=1 && rows!=1)
  {
    VERIFY_RAISES_ASSERT(m1[0]);
    VERIFY_RAISES_ASSERT((m1+m1)[0]);
  }

  VERIFY_IS_APPROX(m3 = m1,m1);
  MatrixType m4;
  VERIFY_IS_APPROX(m4 = m1,m1);

  m3.real() = m1.real();
  VERIFY_IS_APPROX(static_cast<const MatrixType&>(m3).real(), static_cast<const MatrixType&>(m1).real());
  VERIFY_IS_APPROX(static_cast<const MatrixType&>(m3).real(), m1.real());

  // check == / != operators
  VERIFY(m1==m1);
  VERIFY(m1!=m2);
  VERIFY(!(m1==m2));
  VERIFY(!(m1!=m1));
  m1 = m2;
  VERIFY(m1==m2);
  VERIFY(!(m1!=m2));

  // check automatic transposition
  sm2.setZero();
  for(Index i=0;i<rows;++i)
    sm2.col(i) = sm1.row(i);
  VERIFY_IS_APPROX(sm2,sm1.transpose());

  sm2.setZero();
  for(Index i=0;i<rows;++i)
    sm2.col(i).noalias() = sm1.row(i);
  VERIFY_IS_APPROX(sm2,sm1.transpose());

  sm2.setZero();
  for(Index i=0;i<rows;++i)
    sm2.col(i).noalias() += sm1.row(i);
  VERIFY_IS_APPROX(sm2,sm1.transpose());

  sm2.setZero();
  for(Index i=0;i<rows;++i)
    sm2.col(i).noalias() -= sm1.row(i);
  VERIFY_IS_APPROX(sm2,-sm1.transpose());

  // check ternary usage
  {
    bool b = internal::random<int>(0,10)>5;
    m3 = b ? m1 : m2;
    if(b) VERIFY_IS_APPROX(m3,m1);
    else  VERIFY_IS_APPROX(m3,m2);
    m3 = b ? -m1 : m2;
    if(b) VERIFY_IS_APPROX(m3,-m1);
    else  VERIFY_IS_APPROX(m3,m2);
    m3 = b ? m1 : -m2;
    if(b) VERIFY_IS_APPROX(m3,m1);
    else  VERIFY_IS_APPROX(m3,-m2);
  }
}

template<typename MatrixType> void basicStuffComplex(const MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<RealScalar, MatrixType::RowsAtCompileTime, MatrixType::ColsAtCompileTime> RealMatrixType;

  Index rows = m.rows();
  Index cols = m.cols();

  Scalar s1 = internal::random<Scalar>(),
         s2 = internal::random<Scalar>();

  VERIFY(numext::real(s1)==numext::real_ref(s1));
  VERIFY(numext::imag(s1)==numext::imag_ref(s1));
  numext::real_ref(s1) = numext::real(s2);
  numext::imag_ref(s1) = numext::imag(s2);
  VERIFY(internal::isApprox(s1, s2, NumTraits<RealScalar>::epsilon()));
  // extended precision in Intel FPUs means that s1 == s2 in the line above is not guaranteed.

  RealMatrixType rm1 = RealMatrixType::Random(rows,cols),
                 rm2 = RealMatrixType::Random(rows,cols);
  MatrixType cm(rows,cols);
  cm.real() = rm1;
  cm.imag() = rm2;
  VERIFY_IS_APPROX(static_cast<const MatrixType&>(cm).real(), rm1);
  VERIFY_IS_APPROX(static_cast<const MatrixType&>(cm).imag(), rm2);
  rm1.setZero();
  rm2.setZero();
  rm1 = cm.real();
  rm2 = cm.imag();
  VERIFY_IS_APPROX(static_cast<const MatrixType&>(cm).real(), rm1);
  VERIFY_IS_APPROX(static_cast<const MatrixType&>(cm).imag(), rm2);
  cm.real().setZero();
  VERIFY(static_cast<const MatrixType&>(cm).real().isZero());
  VERIFY(!static_cast<const MatrixType&>(cm).imag().isZero());
}

template<typename SrcScalar, typename TgtScalar, bool SrcIsHalfOrBF16 = (internal::is_same<SrcScalar, half>::value || internal::is_same<SrcScalar, bfloat16>::value)> struct casting_test;


template<typename SrcScalar, typename TgtScalar>
struct casting_test<SrcScalar, TgtScalar, false> {
  static void run() {
    Matrix<SrcScalar,4,4> m;
    for (int i=0; i<m.rows(); ++i) {
      for (int j=0; j<m.cols(); ++j) {
        m(i, j) = internal::random_without_cast_overflow<SrcScalar,TgtScalar>::value();
      }
    }
    Matrix<TgtScalar,4,4> n = m.template cast<TgtScalar>();
    for (int i=0; i<m.rows(); ++i) {
      for (int j=0; j<m.cols(); ++j) {
        VERIFY_IS_APPROX(n(i, j), static_cast<TgtScalar>(m(i, j)));
      }
    }
  }
};

template<typename SrcScalar, typename TgtScalar>
struct casting_test<SrcScalar, TgtScalar, true> {
  static void run() {
    casting_test<SrcScalar, TgtScalar, false>::run();
  }
};

template<typename SrcScalar, typename RealScalar>
struct casting_test<SrcScalar, std::complex<RealScalar>, true> {
  static void run() {
    typedef std::complex<RealScalar> TgtScalar;
    Matrix<SrcScalar,4,4> m;
    for (int i=0; i<m.rows(); ++i) {
      for (int j=0; j<m.cols(); ++j) {
        m(i, j) = internal::random_without_cast_overflow<SrcScalar, TgtScalar>::value();
      }
    }
    Matrix<TgtScalar,4,4> n = m.template cast<TgtScalar>();
    for (int i=0; i<m.rows(); ++i) {
      for (int j=0; j<m.cols(); ++j) {
        VERIFY_IS_APPROX(n(i, j), static_cast<TgtScalar>(static_cast<RealScalar>(m(i, j))));
      }
    }
  }
};

template<typename SrcScalar, typename EnableIf = void>
struct casting_test_runner {
  static void run() {
    casting_test<SrcScalar, bool>::run();
    casting_test<SrcScalar, int8_t>::run();
    casting_test<SrcScalar, uint8_t>::run();
    casting_test<SrcScalar, int16_t>::run();
    casting_test<SrcScalar, uint16_t>::run();
    casting_test<SrcScalar, int32_t>::run();
    casting_test<SrcScalar, uint32_t>::run();
#if EIGEN_HAS_CXX11
    casting_test<SrcScalar, int64_t>::run();
    casting_test<SrcScalar, uint64_t>::run();
#endif
    casting_test<SrcScalar, half>::run();
    casting_test<SrcScalar, bfloat16>::run();
    casting_test<SrcScalar, float>::run();
    casting_test<SrcScalar, double>::run();
    casting_test<SrcScalar, std::complex<float> >::run();
    casting_test<SrcScalar, std::complex<double> >::run();
  }
};

template<typename SrcScalar>
struct casting_test_runner<SrcScalar, typename internal::enable_if<(NumTraits<SrcScalar>::IsComplex)>::type>
{
  static void run() {
    // Only a few casts from std::complex<T> are defined.
    casting_test<SrcScalar, half>::run();
    casting_test<SrcScalar, bfloat16>::run();
    casting_test<SrcScalar, std::complex<float> >::run();
    casting_test<SrcScalar, std::complex<double> >::run();
  }
};

void casting_all() {
  casting_test_runner<bool>::run();
  casting_test_runner<int8_t>::run();
  casting_test_runner<uint8_t>::run();
  casting_test_runner<int16_t>::run();
  casting_test_runner<uint16_t>::run();
  casting_test_runner<int32_t>::run();
  casting_test_runner<uint32_t>::run();
#if EIGEN_HAS_CXX11
  casting_test_runner<int64_t>::run();
  casting_test_runner<uint64_t>::run();
#endif
  casting_test_runner<half>::run();
  casting_test_runner<bfloat16>::run();
  casting_test_runner<float>::run();
  casting_test_runner<double>::run();
  casting_test_runner<std::complex<float> >::run();
  casting_test_runner<std::complex<double> >::run();
}

template <typename Scalar>
void fixedSizeMatrixConstruction()
{
  Scalar raw[4];
  for(int k=0; k<4; ++k)
    raw[k] = internal::random<Scalar>();

  {
    Matrix<Scalar,4,1> m(raw);
    Array<Scalar,4,1> a(raw);
    for(int k=0; k<4; ++k) VERIFY(m(k) == raw[k]);
    for(int k=0; k<4; ++k) VERIFY(a(k) == raw[k]);
    VERIFY_IS_EQUAL(m,(Matrix<Scalar,4,1>(raw[0],raw[1],raw[2],raw[3])));
    VERIFY((a==(Array<Scalar,4,1>(raw[0],raw[1],raw[2],raw[3]))).all());
  }
  {
    Matrix<Scalar,3,1> m(raw);
    Array<Scalar,3,1> a(raw);
    for(int k=0; k<3; ++k) VERIFY(m(k) == raw[k]);
    for(int k=0; k<3; ++k) VERIFY(a(k) == raw[k]);
    VERIFY_IS_EQUAL(m,(Matrix<Scalar,3,1>(raw[0],raw[1],raw[2])));
    VERIFY((a==Array<Scalar,3,1>(raw[0],raw[1],raw[2])).all());
  }
  {
    Matrix<Scalar,2,1> m(raw), m2( (DenseIndex(raw[0])), (DenseIndex(raw[1])) );
    Array<Scalar,2,1> a(raw),  a2( (DenseIndex(raw[0])), (DenseIndex(raw[1])) );
    for(int k=0; k<2; ++k) VERIFY(m(k) == raw[k]);
    for(int k=0; k<2; ++k) VERIFY(a(k) == raw[k]);
    VERIFY_IS_EQUAL(m,(Matrix<Scalar,2,1>(raw[0],raw[1])));
    VERIFY((a==Array<Scalar,2,1>(raw[0],raw[1])).all());
    for(int k=0; k<2; ++k) VERIFY(m2(k) == DenseIndex(raw[k]));
    for(int k=0; k<2; ++k) VERIFY(a2(k) == DenseIndex(raw[k]));
  }
  {
    Matrix<Scalar,1,2> m(raw),
                       m2( (DenseIndex(raw[0])), (DenseIndex(raw[1])) ),
                       m3( (int(raw[0])), (int(raw[1])) ),
                       m4( (float(raw[0])), (float(raw[1])) );
    Array<Scalar,1,2> a(raw),  a2( (DenseIndex(raw[0])), (DenseIndex(raw[1])) );
    for(int k=0; k<2; ++k) VERIFY(m(k) == raw[k]);
    for(int k=0; k<2; ++k) VERIFY(a(k) == raw[k]);
    VERIFY_IS_EQUAL(m,(Matrix<Scalar,1,2>(raw[0],raw[1])));
    VERIFY((a==Array<Scalar,1,2>(raw[0],raw[1])).all());
    for(int k=0; k<2; ++k) VERIFY(m2(k) == DenseIndex(raw[k]));
    for(int k=0; k<2; ++k) VERIFY(a2(k) == DenseIndex(raw[k]));
    for(int k=0; k<2; ++k) VERIFY(m3(k) == int(raw[k]));
    for(int k=0; k<2; ++k) VERIFY((m4(k)) == Scalar(float(raw[k])));
  }
  {
    Matrix<Scalar,1,1> m(raw), m1(raw[0]), m2( (DenseIndex(raw[0])) ), m3( (int(raw[0])) );
    Array<Scalar,1,1> a(raw), a1(raw[0]), a2( (DenseIndex(raw[0])) );
    VERIFY(m(0) == raw[0]);
    VERIFY(a(0) == raw[0]);
    VERIFY(m1(0) == raw[0]);
    VERIFY(a1(0) == raw[0]);
    VERIFY(m2(0) == DenseIndex(raw[0]));
    VERIFY(a2(0) == DenseIndex(raw[0]));
    VERIFY(m3(0) == int(raw[0]));
    VERIFY_IS_EQUAL(m,(Matrix<Scalar,1,1>(raw[0])));
    VERIFY((a==Array<Scalar,1,1>(raw[0])).all());
  }
}

EIGEN_DECLARE_TEST(basicstuff)
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( basicStuff(Matrix<float, 1, 1>()) );
    CALL_SUBTEST_2( basicStuff(Matrix4d()) );
    CALL_SUBTEST_3( basicStuff(MatrixXcf(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_4( basicStuff(MatrixXi(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_5( basicStuff(MatrixXcd(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_6( basicStuff(Matrix<float, 100, 100>()) );
    CALL_SUBTEST_7( basicStuff(Matrix<long double,Dynamic,Dynamic>(internal::random<int>(1,EIGEN_TEST_MAX_SIZE),internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_8( casting_all() );

    CALL_SUBTEST_3( basicStuffComplex(MatrixXcf(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_5( basicStuffComplex(MatrixXcd(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
  }

  CALL_SUBTEST_1(fixedSizeMatrixConstruction<unsigned char>());
  CALL_SUBTEST_1(fixedSizeMatrixConstruction<float>());
  CALL_SUBTEST_1(fixedSizeMatrixConstruction<double>());
  CALL_SUBTEST_1(fixedSizeMatrixConstruction<int>());
  CALL_SUBTEST_1(fixedSizeMatrixConstruction<long int>());
  CALL_SUBTEST_1(fixedSizeMatrixConstruction<std::ptrdiff_t>());
}
