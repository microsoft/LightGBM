/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#pragma once

#ifndef FREEFORM2_ARRAY_DEREFERENCE_EXPRESSION_H
#define FREEFORM2_ARRAY_DEREFERENCE_EXPRESSION_H

#include "Expression.h"

namespace FreeForm2 {
class ArrayType;
class ProgramParseState;

// An array-dereference expression removes a dimension from an array.
class ArrayDereferenceExpression : public Expression {
 public:
  ArrayDereferenceExpression(const Annotations &p_annotations,
                             const Expression &p_array,
                             const Expression &p_index, size_t p_version);

  // Methods inherited from Expression.
  virtual size_t GetNumChildren() const override;
  virtual void Accept(Visitor &p_visitor) const override;
  virtual void AcceptReference(Visitor &p_visitor) const override;
  virtual const TypeImpl &GetType() const override;

  const Expression &GetArray() const;
  const Expression &GetIndex() const;
  size_t GetVersion() const;

  // Gets the VariableID of the array object, regardless of the number
  // of dereferences.
  VariableID GetBaseArrayId() const;

 private:
  // Dereferenced type of this expression.
  const TypeImpl &m_type;

  // Array expression being dereferenced.
  const Expression &m_array;

  // Index supplied.
  const Expression &m_index;

  // A unique version number associated with a particular
  // value for this variable.
  const size_t m_version;
};
};  // namespace FreeForm2

#endif
