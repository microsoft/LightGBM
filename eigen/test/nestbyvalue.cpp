// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2019 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define TEST_ENABLE_TEMPORARY_TRACKING

#include "main.h"

typedef NestByValue<MatrixXd> CpyMatrixXd;
typedef CwiseBinaryOp<internal::scalar_sum_op<double,double>,const CpyMatrixXd,const CpyMatrixXd> XprType;

XprType get_xpr_with_temps(const MatrixXd& a)
{
  MatrixXd t1 = a.rowwise().reverse();
  MatrixXd t2 = a+a;
  return t1.nestByValue() + t2.nestByValue();
}

EIGEN_DECLARE_TEST(nestbyvalue)
{
  for(int i = 0; i < g_repeat; i++) {
    Index rows = internal::random<Index>(1,EIGEN_TEST_MAX_SIZE);
    Index cols = internal::random<Index>(1,EIGEN_TEST_MAX_SIZE);
    MatrixXd a = MatrixXd(rows,cols);
    nb_temporaries = 0;
    XprType x = get_xpr_with_temps(a);
    VERIFY_IS_EQUAL(nb_temporaries,6);
    MatrixXd b = x;
    VERIFY_IS_EQUAL(nb_temporaries,6+1);
    VERIFY_IS_APPROX(b, a.rowwise().reverse().eval() + (a+a).eval());
  }
}
