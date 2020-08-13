// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <g.gael@free.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_NO_DEBUG_SMALL_PRODUCT_BLOCKS
#include "sparse_solver.h"

#include <Eigen/KLUSupport>

template<typename T> void test_klu_support_T()
{
  KLU<SparseMatrix<T, ColMajor> > klu_colmajor;
  KLU<SparseMatrix<T, RowMajor> > klu_rowmajor;
  
  check_sparse_square_solving(klu_colmajor);
  check_sparse_square_solving(klu_rowmajor);
  
  //check_sparse_square_determinant(umfpack_colmajor);
  //check_sparse_square_determinant(umfpack_rowmajor);
}

EIGEN_DECLARE_TEST(klu_support)
{
  CALL_SUBTEST_1(test_klu_support_T<double>());
  CALL_SUBTEST_2(test_klu_support_T<std::complex<double> >());
}

