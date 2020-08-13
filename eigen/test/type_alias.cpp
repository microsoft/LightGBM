// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2019 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

EIGEN_DECLARE_TEST(type_alias)
{
  using namespace internal;

  // To warm up, some basic checks:
  STATIC_CHECK((is_same<MatrixXd,Matrix<double,Dynamic,Dynamic> >::value));
  STATIC_CHECK((is_same<Matrix2f,Matrix<float,2,2> >::value));
  STATIC_CHECK((is_same<Array33i,Array<int,3,3> >::value));

#if EIGEN_HAS_CXX11
  
  STATIC_CHECK((is_same<MatrixX<double>,    MatrixXd>::value));
  STATIC_CHECK((is_same<MatrixX<int>,       MatrixXi>::value));
  STATIC_CHECK((is_same<Matrix2<int>,       Matrix2i>::value));
  STATIC_CHECK((is_same<Matrix2X<float>,    Matrix2Xf>::value));
  STATIC_CHECK((is_same<MatrixX4<double>,   MatrixX4d>::value));
  STATIC_CHECK((is_same<VectorX<int>,       VectorXi>::value));
  STATIC_CHECK((is_same<Vector2<float>,     Vector2f>::value));
  STATIC_CHECK((is_same<RowVectorX<int>,    RowVectorXi>::value));
  STATIC_CHECK((is_same<RowVector2<float>,  RowVector2f>::value));

  STATIC_CHECK((is_same<ArrayXX<float>,     ArrayXXf>::value));
  STATIC_CHECK((is_same<Array33<int>,       Array33i>::value));
  STATIC_CHECK((is_same<Array2X<float>,     Array2Xf>::value));
  STATIC_CHECK((is_same<ArrayX4<double>,    ArrayX4d>::value));
  STATIC_CHECK((is_same<ArrayX<double>,     ArrayXd>::value));
  STATIC_CHECK((is_same<Array4<double>,     Array4d>::value));

  STATIC_CHECK((is_same<Vector<float,3>,        Vector3f>::value));
  STATIC_CHECK((is_same<Vector<int,Dynamic>,    VectorXi>::value));
  STATIC_CHECK((is_same<RowVector<float,3>,     RowVector3f>::value));
  STATIC_CHECK((is_same<RowVector<int,Dynamic>, RowVectorXi>::value));

#else
  std::cerr << "WARNING: c++11 type aliases not tested.\n";
#endif
}
