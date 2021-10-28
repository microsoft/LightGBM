/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#pragma once

#ifndef FREEFORM2_SELECT_NTH_H
#define FREEFORM2_SELECT_NTH_H

#include <boost/shared_ptr.hpp>
#include <vector>

#include "Expression.h"

namespace DynamicRank {
class IFeatureMap;
class INeuralNetFeatures;
}  // namespace DynamicRank

namespace FreeForm2 {
class TypeManager;
class ArrayType;

// SelectNth is a simple expression to which the programmer provides an
// index (first arg), and then a series of values.  The index selects a
// value based on ordinal position (0 is first, 1 second, etc).  If the
// programmer provides an out-of-bounds index, we provide the value from the
// nearest available index (either lowest or highest, depending on which
// side the programmer fell off the end of the expression list).
class SelectNthExpression : public Expression {
 public:
  // Methods inherited from Expression.
  virtual void Accept(Visitor &p_visitor) const override;
  virtual size_t GetNumChildren() const override;
  virtual const TypeImpl &GetType() const override;

  const Expression &GetIndex() const;
  const Expression &GetChild(size_t p_index) const;

  static boost::shared_ptr<SelectNthExpression> Alloc(
      const Annotations &p_annotations,
      const std::vector<const Expression *> &p_children);

 private:
  SelectNthExpression(const Annotations &p_annotations,
                      const std::vector<const Expression *> &p_children);

  static void DeleteAlloc(SelectNthExpression *p_allocated);

  // Infer the result type of the select-nth expression.
  const TypeImpl &InferType() const;

  // The type of this expression.
  const TypeImpl *m_type;

  // Integer expression giving the selection index.
  const Expression &m_index;

  // Number of children stored in m_numChildren.
  unsigned int m_numChildren;

  // Array of children of this node, allocated using struct hack.
  const Expression *m_children[1];
};

// SelectRange is similar in concept to SelectNth: it selects a slice of
// an array. The expression takes a start index, a count, and an array. It
// evaluates to an array which has the same content as the source array
// slice starting at the start index, with count elements. If the start of
// the slice is negative, an empty array is returned (regardless of size).
// The count will be limited such that this expression will produce the
// largest array possible containing elements from the source. A negative
// or zero count will return an empty array.
class SelectRangeExpression : public Expression {
 public:
  SelectRangeExpression(const Annotations &p_annotations,
                        const Expression &p_start, const Expression &p_count,
                        const Expression &p_array, TypeManager &p_typeManager);

  // Methods inherited from Expression.
  virtual void Accept(Visitor &p_visitor) const override;
  virtual size_t GetNumChildren() const override;
  virtual const TypeImpl &GetType() const override;

  // Accessor methods for the properties of this class.
  const Expression &GetStart() const;
  const Expression &GetCount() const;
  const Expression &GetArray() const;

 private:
  // The type of this expression.
  const ArrayType *m_type;

  // Integer expression giving the start of the slice.
  const Expression &m_start;

  // Integer expression giving the count of elements in the slice.
  const Expression &m_count;

  // The array from which a slice is being taken.
  const Expression &m_array;
};
};  // namespace FreeForm2

#endif
