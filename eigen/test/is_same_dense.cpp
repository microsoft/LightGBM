// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

using internal::is_same_dense;

EIGEN_DECLARE_TEST(is_same_dense)
{
  typedef Matrix<double,Dynamic,Dynamic,ColMajor> ColMatrixXd;
  typedef Matrix<std::complex<double>,Dynamic,Dynamic,ColMajor> ColMatrixXcd;
  ColMatrixXd m1(10,10);
  ColMatrixXcd m2(10,10);
  Ref<ColMatrixXd> ref_m1(m1);
  Ref<ColMatrixXd,0, Stride<Dynamic,Dynamic> >  ref_m2_real(m2.real());
  Ref<const ColMatrixXd> const_ref_m1(m1);

  VERIFY(is_same_dense(m1,m1));
  VERIFY(is_same_dense(m1,ref_m1));
  VERIFY(is_same_dense(const_ref_m1,m1));
  VERIFY(is_same_dense(const_ref_m1,ref_m1));
  
  VERIFY(is_same_dense(m1.block(0,0,m1.rows(),m1.cols()),m1));
  VERIFY(!is_same_dense(m1.row(0),m1.col(0)));
  
  Ref<const ColMatrixXd> const_ref_m1_row(m1.row(1));
  VERIFY(!is_same_dense(m1.row(1),const_ref_m1_row));
  
  Ref<const ColMatrixXd> const_ref_m1_col(m1.col(1));
  VERIFY(is_same_dense(m1.col(1),const_ref_m1_col));


  VERIFY(!is_same_dense(m1, ref_m2_real));
  VERIFY(!is_same_dense(m2, ref_m2_real));
}
