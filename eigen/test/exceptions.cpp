// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


// Various sanity tests with exceptions and non trivially copyable scalar type.
//  - no memory leak when a custom scalar type trow an exceptions
//  - todo: complete the list of tests!

#define EIGEN_STACK_ALLOCATION_LIMIT 100000000

#include "main.h"
#include "AnnoyingScalar.h"

#define CHECK_MEMLEAK(OP) {                                 \
    AnnoyingScalar::countdown = 100;                        \
    int before = AnnoyingScalar::instances;                 \
    bool exception_thrown = false;                          \
    try { OP; }                                             \
    catch (my_exception) {                                  \
      exception_thrown = true;                              \
      VERIFY(AnnoyingScalar::instances==before && "memory leak detected in " && EIGEN_MAKESTRING(OP)); \
    } \
    VERIFY( (AnnoyingScalar::dont_throw) || (exception_thrown && " no exception thrown in " && EIGEN_MAKESTRING(OP)) ); \
  }

EIGEN_DECLARE_TEST(exceptions)
{
  typedef Eigen::Matrix<AnnoyingScalar,Dynamic,1> VectorType;
  typedef Eigen::Matrix<AnnoyingScalar,Dynamic,Dynamic> MatrixType;
  
  {
    AnnoyingScalar::dont_throw = false;
    int n = 50;
    VectorType v0(n), v1(n);
    MatrixType m0(n,n), m1(n,n), m2(n,n);
    v0.setOnes(); v1.setOnes();
    m0.setOnes(); m1.setOnes(); m2.setOnes();
    CHECK_MEMLEAK(v0 = m0 * m1 * v1);
    CHECK_MEMLEAK(m2 = m0 * m1 * m2);
    CHECK_MEMLEAK((v0+v1).dot(v0+v1));
  }
  VERIFY(AnnoyingScalar::instances==0 && "global memory leak detected in " && EIGEN_MAKESTRING(OP));
}
