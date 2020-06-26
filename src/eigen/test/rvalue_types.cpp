// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2013 Hauke Heibel <hauke.heibel@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#include <Eigen/Core>

using internal::UIntPtr;

#if EIGEN_HAS_RVALUE_REFERENCES
template <typename MatrixType>
void rvalue_copyassign(const MatrixType& m)
{

  typedef typename internal::traits<MatrixType>::Scalar Scalar;
  
  // create a temporary which we are about to destroy by moving
  MatrixType tmp = m;
  UIntPtr src_address = reinterpret_cast<UIntPtr>(tmp.data());
  
  // move the temporary to n
  MatrixType n = std::move(tmp);
  UIntPtr dst_address = reinterpret_cast<UIntPtr>(n.data());

  if (MatrixType::RowsAtCompileTime==Dynamic|| MatrixType::ColsAtCompileTime==Dynamic)
  {
    // verify that we actually moved the guts
    VERIFY_IS_EQUAL(src_address, dst_address);
  }

  // verify that the content did not change
  Scalar abs_diff = (m-n).array().abs().sum();
  VERIFY_IS_EQUAL(abs_diff, Scalar(0));
}
#else
template <typename MatrixType>
void rvalue_copyassign(const MatrixType&) {}
#endif

void test_rvalue_types()
{
  CALL_SUBTEST_1(rvalue_copyassign( MatrixXf::Random(50,50).eval() ));
  CALL_SUBTEST_1(rvalue_copyassign( ArrayXXf::Random(50,50).eval() ));

  CALL_SUBTEST_1(rvalue_copyassign( Matrix<float,1,Dynamic>::Random(50).eval() ));
  CALL_SUBTEST_1(rvalue_copyassign( Array<float,1,Dynamic>::Random(50).eval() ));

  CALL_SUBTEST_1(rvalue_copyassign( Matrix<float,Dynamic,1>::Random(50).eval() ));
  CALL_SUBTEST_1(rvalue_copyassign( Array<float,Dynamic,1>::Random(50).eval() ));
  
  CALL_SUBTEST_2(rvalue_copyassign( Array<float,2,1>::Random().eval() ));
  CALL_SUBTEST_2(rvalue_copyassign( Array<float,3,1>::Random().eval() ));
  CALL_SUBTEST_2(rvalue_copyassign( Array<float,4,1>::Random().eval() ));

  CALL_SUBTEST_2(rvalue_copyassign( Array<float,2,2>::Random().eval() ));
  CALL_SUBTEST_2(rvalue_copyassign( Array<float,3,3>::Random().eval() ));
  CALL_SUBTEST_2(rvalue_copyassign( Array<float,4,4>::Random().eval() ));
}
