/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#pragma once

#ifndef FREEFORM2_RESULT_H
#define FREEFORM2_RESULT_H

#include <basic_types.h>

#include <boost/iterator/iterator_facade.hpp>
#include <boost/noncopyable.hpp>
#include <iostream>
#include <memory>

#include "FreeForm2Type.h"

namespace FreeForm2 {
// Class to encapsulate value and type into a result.
class ResultIterator;
class Result : boost::noncopyable {
 public:
  typedef bool BoolType;
  typedef Int64 IntType;
  typedef UInt64 UInt64Type;
  typedef Int32 Int32Type;
  typedef UInt32 UInt32Type;

  // Generate minimum int for this size via bit twiddling.
  static const IntType c_minInt =
      (static_cast<IntType>(1) << ((sizeof(IntType) * 8) - 1));

  // Generate maximum int for this size by underflowing from c_minInt.
  // (disable warning about integer overflow).
#pragma warning(push)
#pragma warning(disable : 4307)
  static const IntType c_maxInt = c_minInt - 1;
#pragma warning(pop)

  typedef float FloatType;

  virtual ~Result();

  // Compare two results (which must be of the same type, else an
  // exception is thrown), returning the usual c convention of < 0, 0,
  // > 0 to indicate less than, equal to or greater than the other result.
  int Compare(const Result &p_other) const;

  // Print the result to the given output stream.
  void Print(std::ostream &p_out) const;

  virtual const Type &GetType() const = 0;
  virtual IntType GetInt() const = 0;
  virtual UInt64Type GetUInt64() const = 0;
  virtual Int32Type GetInt32() const = 0;
  virtual UInt32Type GetUInt32() const = 0;
  virtual FloatType GetFloat() const = 0;
  virtual BoolType GetBool() const = 0;
  virtual ResultIterator BeginArray() const = 0;
  virtual ResultIterator EndArray() const = 0;

  // Compare two floats for equality, using the freeform2 standard
  // tolerance.
  static int CompareFloat(FloatType p_left, FloatType p_right);
};

// Facade class to help iterate over array elements in an abstract way.
// Note that boost::iterator_facade and virtual functions don't place nicely
// together, so we need a separate virtual class, and an iterator_facade class.
class ResultIteratorImpl;
class ResultIterator
    : public boost::iterator_facade<ResultIterator, const Result,
                                    boost::random_access_traversal_tag> {
  friend class boost::iterator_core_access;

 public:
  explicit ResultIterator(std::auto_ptr<ResultIteratorImpl> p_impl);

  ResultIterator(const ResultIterator &p_other);

  ~ResultIterator();

 private:
  // iterator_facade function to increment the iterator.
  void increment();

  // iterator_facade function to decrement the iterator.
  void decrement();

  // iterator_facade function to compare iterators.
  bool equal(const ResultIterator &p_other) const;

  // iterator_facade function to get the current element.
  const Result &dereference() const;

  // iterator_facade function to get the current element.
  void advance(std::ptrdiff_t p_distance);

  // iterator_facade function to calculate distance to another iterator.
  std::ptrdiff_t distance_to(const ResultIterator &p_other) const;

  // Pointer to virtual iterator implementation.
  std::auto_ptr<ResultIteratorImpl> m_impl;
};

std::ostream &operator<<(std::ostream &p_out, const Result &p_result);

// This union defines the various compile-time constant values in the
// compiler.
union ConstantValue {
  Result::IntType m_int;
  Result::UInt64Type m_uint64;
  Result::Int32Type m_int32;
  Result::UInt32Type m_uint32;
  Result::FloatType m_float;
  Result::BoolType m_bool;

  Result::IntType GetInt(const TypeImpl &p_type) const;
};

}  // namespace FreeForm2

#endif
