// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2020 Sebastien Boisvert <seb@boisvert.info>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "BenchTimer.h"
#include "../test/MovableScalar.h"

#include <Eigen/Core>

#include <iostream>
#include <utility>

template <typename MatrixType>
void copy_matrix(MatrixType& m)
{
  MatrixType tmp(m);
  m = tmp;
}

template <typename MatrixType>
void move_matrix(MatrixType&& m)
{
  MatrixType tmp(std::move(m));
  m = std::move(tmp);
}

template<typename Scalar>
void bench(const std::string& label)
{
  using MatrixType = Eigen::Matrix<Eigen::MovableScalar<Scalar>,1,10>;
  Eigen::BenchTimer t;

  int tries = 10;
  int rep = 1000000;

  MatrixType data = MatrixType::Random().eval();
  MatrixType dest;

  BENCH(t, tries, rep, copy_matrix(data));
  std::cout << label << " copy semantics: " << 1e3*t.best(Eigen::CPU_TIMER) << " ms" << std::endl;

  BENCH(t, tries, rep, move_matrix(std::move(data)));
  std::cout << label << " move semantics: " << 1e3*t.best(Eigen::CPU_TIMER) << " ms" << std::endl;
}

int main()
{
  bench<float>("float");
  bench<double>("double");
  return 0;
}

