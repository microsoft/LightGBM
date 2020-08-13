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

#include <Eigen/CholmodSupport>

template<typename SparseType> void test_cholmod_ST()
{
  CholmodDecomposition<SparseType, Lower> g_chol_colmajor_lower; g_chol_colmajor_lower.setMode(CholmodSupernodalLLt);
  CholmodDecomposition<SparseType, Upper> g_chol_colmajor_upper; g_chol_colmajor_upper.setMode(CholmodSupernodalLLt);
  CholmodDecomposition<SparseType, Lower> g_llt_colmajor_lower;  g_llt_colmajor_lower.setMode(CholmodSimplicialLLt);
  CholmodDecomposition<SparseType, Upper> g_llt_colmajor_upper;  g_llt_colmajor_upper.setMode(CholmodSimplicialLLt);
  CholmodDecomposition<SparseType, Lower> g_ldlt_colmajor_lower; g_ldlt_colmajor_lower.setMode(CholmodLDLt);
  CholmodDecomposition<SparseType, Upper> g_ldlt_colmajor_upper; g_ldlt_colmajor_upper.setMode(CholmodLDLt);
  
  CholmodSupernodalLLT<SparseType, Lower> chol_colmajor_lower;
  CholmodSupernodalLLT<SparseType, Upper> chol_colmajor_upper;
  CholmodSimplicialLLT<SparseType, Lower> llt_colmajor_lower;
  CholmodSimplicialLLT<SparseType, Upper> llt_colmajor_upper;
  CholmodSimplicialLDLT<SparseType, Lower> ldlt_colmajor_lower;
  CholmodSimplicialLDLT<SparseType, Upper> ldlt_colmajor_upper;

  check_sparse_spd_solving(g_chol_colmajor_lower);
  check_sparse_spd_solving(g_chol_colmajor_upper);
  check_sparse_spd_solving(g_llt_colmajor_lower);
  check_sparse_spd_solving(g_llt_colmajor_upper);
  check_sparse_spd_solving(g_ldlt_colmajor_lower);
  check_sparse_spd_solving(g_ldlt_colmajor_upper);
  
  check_sparse_spd_solving(chol_colmajor_lower);
  check_sparse_spd_solving(chol_colmajor_upper);
  check_sparse_spd_solving(llt_colmajor_lower);
  check_sparse_spd_solving(llt_colmajor_upper);
  check_sparse_spd_solving(ldlt_colmajor_lower);
  check_sparse_spd_solving(ldlt_colmajor_upper);

  check_sparse_spd_determinant(chol_colmajor_lower);
  check_sparse_spd_determinant(chol_colmajor_upper);
  check_sparse_spd_determinant(llt_colmajor_lower);
  check_sparse_spd_determinant(llt_colmajor_upper);
  check_sparse_spd_determinant(ldlt_colmajor_lower);
  check_sparse_spd_determinant(ldlt_colmajor_upper);
}

template<typename T, int flags, typename IdxType> void test_cholmod_T()
{
    test_cholmod_ST<SparseMatrix<T, flags, IdxType> >();
}

EIGEN_DECLARE_TEST(cholmod_support)
{
  CALL_SUBTEST_11( (test_cholmod_T<double              , ColMajor, int >()) );
  CALL_SUBTEST_12( (test_cholmod_T<double              , ColMajor, long>()) );
  CALL_SUBTEST_13( (test_cholmod_T<double              , RowMajor, int >()) );
  CALL_SUBTEST_14( (test_cholmod_T<double              , RowMajor, long>()) );
  CALL_SUBTEST_21( (test_cholmod_T<std::complex<double>, ColMajor, int >()) );
  CALL_SUBTEST_22( (test_cholmod_T<std::complex<double>, ColMajor, long>()) );
  // TODO complex row-major matrices do not work at the moment:
  // CALL_SUBTEST_23( (test_cholmod_T<std::complex<double>, RowMajor, int >()) );
  // CALL_SUBTEST_24( (test_cholmod_T<std::complex<double>, RowMajor, long>()) );
}
