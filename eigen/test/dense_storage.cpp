// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2013 Hauke Heibel <hauke.heibel@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#include <Eigen/Core>

template <typename T, int Rows, int Cols>
void dense_storage_copy()
{
  static const int Size = ((Rows==Dynamic || Cols==Dynamic) ? Dynamic : Rows*Cols);
  typedef DenseStorage<T,Size, Rows,Cols, 0> DenseStorageType;
  
  const int rows = (Rows==Dynamic) ? 4 : Rows;
  const int cols = (Cols==Dynamic) ? 3 : Cols;
  const int size = rows*cols;
  DenseStorageType reference(size, rows, cols);
  T* raw_reference = reference.data();
  for (int i=0; i<size; ++i)
    raw_reference[i] = static_cast<T>(i);
    
  DenseStorageType copied_reference(reference);
  const T* raw_copied_reference = copied_reference.data();
  for (int i=0; i<size; ++i)
    VERIFY_IS_EQUAL(raw_reference[i], raw_copied_reference[i]);
}

template <typename T, int Rows, int Cols>
void dense_storage_assignment()
{
  static const int Size = ((Rows==Dynamic || Cols==Dynamic) ? Dynamic : Rows*Cols);
  typedef DenseStorage<T,Size, Rows,Cols, 0> DenseStorageType;
  
  const int rows = (Rows==Dynamic) ? 4 : Rows;
  const int cols = (Cols==Dynamic) ? 3 : Cols;
  const int size = rows*cols;
  DenseStorageType reference(size, rows, cols);
  T* raw_reference = reference.data();
  for (int i=0; i<size; ++i)
    raw_reference[i] = static_cast<T>(i);
    
  DenseStorageType copied_reference;
  copied_reference = reference;
  const T* raw_copied_reference = copied_reference.data();
  for (int i=0; i<size; ++i)
    VERIFY_IS_EQUAL(raw_reference[i], raw_copied_reference[i]);
}

template<typename T, int Size, std::size_t Alignment>
void dense_storage_alignment()
{
  #if EIGEN_HAS_ALIGNAS
  
  struct alignas(Alignment) Empty1 {};
  VERIFY_IS_EQUAL(std::alignment_of<Empty1>::value, Alignment);

  struct EIGEN_ALIGN_TO_BOUNDARY(Alignment) Empty2 {};
  VERIFY_IS_EQUAL(std::alignment_of<Empty2>::value, Alignment);

  struct Nested1 { EIGEN_ALIGN_TO_BOUNDARY(Alignment) T data[Size]; };
  VERIFY_IS_EQUAL(std::alignment_of<Nested1>::value, Alignment);

  VERIFY_IS_EQUAL( (std::alignment_of<internal::plain_array<T,Size,AutoAlign,Alignment> >::value), Alignment);

  const std::size_t default_alignment = internal::compute_default_alignment<T,Size>::value;

  VERIFY_IS_EQUAL( (std::alignment_of<DenseStorage<T,Size,1,1,AutoAlign> >::value), default_alignment);
  VERIFY_IS_EQUAL( (std::alignment_of<Matrix<T,Size,1,AutoAlign> >::value), default_alignment);
  struct Nested2 { Matrix<T,Size,1,AutoAlign> mat; };
  VERIFY_IS_EQUAL(std::alignment_of<Nested2>::value, default_alignment);

  #endif
}

EIGEN_DECLARE_TEST(dense_storage)
{
  dense_storage_copy<int,Dynamic,Dynamic>();  
  dense_storage_copy<int,Dynamic,3>();
  dense_storage_copy<int,4,Dynamic>();
  dense_storage_copy<int,4,3>();

  dense_storage_copy<float,Dynamic,Dynamic>();
  dense_storage_copy<float,Dynamic,3>();
  dense_storage_copy<float,4,Dynamic>();  
  dense_storage_copy<float,4,3>();
  
  dense_storage_assignment<int,Dynamic,Dynamic>();  
  dense_storage_assignment<int,Dynamic,3>();
  dense_storage_assignment<int,4,Dynamic>();
  dense_storage_assignment<int,4,3>();

  dense_storage_assignment<float,Dynamic,Dynamic>();
  dense_storage_assignment<float,Dynamic,3>();
  dense_storage_assignment<float,4,Dynamic>();  
  dense_storage_assignment<float,4,3>(); 

  dense_storage_alignment<float,16,8>();
  dense_storage_alignment<float,16,16>();
  dense_storage_alignment<float,16,32>();
  dense_storage_alignment<float,16,64>();
}
