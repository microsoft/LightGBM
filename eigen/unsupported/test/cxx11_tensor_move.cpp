// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Viktor Csomor <viktor.csomor@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#include <Eigen/CXX11/Tensor>
#include <utility>

using Eigen::Tensor;
using Eigen::RowMajor;

static void calc_indices(int i, int& x, int& y, int& z)
{
  x = i / 4;
  y = (i % 4) / 2;
  z = i % 2;
}

static void test_move()
{
  int x;
  int y;
  int z;

  Tensor<int,3> tensor1(2, 2, 2);
  Tensor<int,3,RowMajor> tensor2(2, 2, 2);

  for (int i = 0; i < 8; i++)
  {
    calc_indices(i, x, y, z);
    tensor1(x,y,z) = i;
    tensor2(x,y,z) = 2 * i;
  }

  // Invokes the move constructor.
  Tensor<int,3> moved_tensor1 = std::move(tensor1);
  Tensor<int,3,RowMajor> moved_tensor2 = std::move(tensor2);

  VERIFY_IS_EQUAL(tensor1.size(), 0);
  VERIFY_IS_EQUAL(tensor2.size(), 0);

  for (int i = 0; i < 8; i++)
  {
    calc_indices(i, x, y, z);
    VERIFY_IS_EQUAL(moved_tensor1(x,y,z), i);
    VERIFY_IS_EQUAL(moved_tensor2(x,y,z), 2 * i);
  }

  Tensor<int,3> moved_tensor3(2,2,2);
  Tensor<int,3,RowMajor> moved_tensor4(2,2,2);

  moved_tensor3.setZero();
  moved_tensor4.setZero();

  // Invokes the move assignment operator.
  moved_tensor3 = std::move(moved_tensor1);
  moved_tensor4 = std::move(moved_tensor2);

  VERIFY_IS_EQUAL(moved_tensor1.size(), 8);
  VERIFY_IS_EQUAL(moved_tensor2.size(), 8);

  for (int i = 0; i < 8; i++)
  {
    calc_indices(i, x, y, z);
    VERIFY_IS_EQUAL(moved_tensor1(x,y,z), 0);
    VERIFY_IS_EQUAL(moved_tensor2(x,y,z), 0);
    VERIFY_IS_EQUAL(moved_tensor3(x,y,z), i);
    VERIFY_IS_EQUAL(moved_tensor4(x,y,z), 2 * i);
  }
}

EIGEN_DECLARE_TEST(cxx11_tensor_move)
{
  CALL_SUBTEST(test_move());
}
