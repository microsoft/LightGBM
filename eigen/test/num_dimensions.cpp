// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2018 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <Eigen/SparseCore>

template<int ExpectedDim,typename Xpr>
void check_dim(const Xpr& ) {
  STATIC_CHECK( Xpr::NumDimensions == ExpectedDim );
}

#if EIGEN_HAS_CXX11
template<template <typename,int,int> class Object>
void map_num_dimensions()
{
  typedef Object<double, 1, 1> ArrayScalarType;
  typedef Object<double, 2, 1> ArrayVectorType;
  typedef Object<double, 1, 2> TransposeArrayVectorType;
  typedef Object<double, 2, 2> ArrayType;
  typedef Object<double, Eigen::Dynamic, 1> DynamicArrayVectorType;
  typedef Object<double, 1, Eigen::Dynamic> DynamicTransposeArrayVectorType;
  typedef Object<double, Eigen::Dynamic, Eigen::Dynamic> DynamicArrayType;

  STATIC_CHECK(ArrayScalarType::NumDimensions == 0);
  STATIC_CHECK(ArrayVectorType::NumDimensions == 1);
  STATIC_CHECK(TransposeArrayVectorType::NumDimensions == 1);
  STATIC_CHECK(ArrayType::NumDimensions == 2);
  STATIC_CHECK(DynamicArrayVectorType::NumDimensions == 1);
  STATIC_CHECK(DynamicTransposeArrayVectorType::NumDimensions == 1);
  STATIC_CHECK(DynamicArrayType::NumDimensions == 2);

  typedef Eigen::Map<ArrayScalarType> ArrayScalarMap;
  typedef Eigen::Map<ArrayVectorType> ArrayVectorMap;
  typedef Eigen::Map<TransposeArrayVectorType> TransposeArrayVectorMap;
  typedef Eigen::Map<ArrayType> ArrayMap;
  typedef Eigen::Map<DynamicArrayVectorType> DynamicArrayVectorMap;
  typedef Eigen::Map<DynamicTransposeArrayVectorType> DynamicTransposeArrayVectorMap;
  typedef Eigen::Map<DynamicArrayType> DynamicArrayMap;

  STATIC_CHECK(ArrayScalarMap::NumDimensions == 0);
  STATIC_CHECK(ArrayVectorMap::NumDimensions == 1);
  STATIC_CHECK(TransposeArrayVectorMap::NumDimensions == 1);
  STATIC_CHECK(ArrayMap::NumDimensions == 2);
  STATIC_CHECK(DynamicArrayVectorMap::NumDimensions == 1);
  STATIC_CHECK(DynamicTransposeArrayVectorMap::NumDimensions == 1);
  STATIC_CHECK(DynamicArrayMap::NumDimensions == 2);
}

template<typename Scalar, int Rows, int Cols>
using TArray = Array<Scalar,Rows,Cols>;

template<typename Scalar, int Rows, int Cols>
using TMatrix = Matrix<Scalar,Rows,Cols>;

#endif

EIGEN_DECLARE_TEST(num_dimensions)
{
  int n = 10;
  ArrayXXd A(n,n);
  CALL_SUBTEST( check_dim<2>(A) );
  CALL_SUBTEST( check_dim<2>(A.block(1,1,2,2)) );
  CALL_SUBTEST( check_dim<1>(A.col(1)) );
  CALL_SUBTEST( check_dim<1>(A.row(1)) );

  MatrixXd M(n,n);
  CALL_SUBTEST( check_dim<0>(M.row(1)*M.col(1)) );

  SparseMatrix<double> S(n,n);
  CALL_SUBTEST( check_dim<2>(S) );
  CALL_SUBTEST( check_dim<2>(S.block(1,1,2,2)) );
  CALL_SUBTEST( check_dim<1>(S.col(1)) );
  CALL_SUBTEST( check_dim<1>(S.row(1)) );

  SparseVector<double> s(n);
  CALL_SUBTEST( check_dim<1>(s) );
  CALL_SUBTEST( check_dim<1>(s.head(2)) );
  

  #if EIGEN_HAS_CXX11
  CALL_SUBTEST( map_num_dimensions<TArray>() );
  CALL_SUBTEST( map_num_dimensions<TMatrix>() );
  #endif
}
