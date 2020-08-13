// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2020 Sebastien Boisvert <seb@boisvert.info>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MISC_MOVABLE_SCALAR_H
#define EIGEN_MISC_MOVABLE_SCALAR_H

#include <vector>

namespace Eigen
{
template <typename Scalar, typename Base = std::vector<Scalar>>
struct MovableScalar : public Base
{
  MovableScalar() = default;
  ~MovableScalar() = default;
  MovableScalar(const MovableScalar&) = default;
  MovableScalar(MovableScalar&& other) = default;
  MovableScalar& operator=(const MovableScalar&) = default;
  MovableScalar& operator=(MovableScalar&& other) = default;
  MovableScalar(Scalar scalar) : Base(100, scalar) {}

  operator Scalar() const { return this->size() > 0 ? this->back() : Scalar(); }
};

template<> struct NumTraits<MovableScalar<float>> : GenericNumTraits<float> {};
}

#endif

