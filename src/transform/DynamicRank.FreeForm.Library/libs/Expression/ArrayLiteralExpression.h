/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#pragma once

#ifndef FREEFORM2_ARRAY_LITERAL_EXPRESSION_H
#define FREEFORM2_ARRAY_LITERAL_EXPRESSION_H

#include <vector>

#include "ArrayType.h"
#include "Expression.h"

// (array-literal [ ... ]) or (array-literal [ ...] type)

namespace FreeForm2 {
class SimpleExpressionOwner;
class ProgramParseState;
class TypeManager;
class Visitor;

// An array-literal expression generates an array literal.
class ArrayLiteralExpression : public Expression {
 public:
  // Methods inherited from Expression.
  size_t GetNumChildren() const override;
  virtual void Accept(Visitor &p_visitor) const override;
  virtual void AcceptReference(Visitor &p_visitor) const override;
  virtual const TypeImpl &GetType() const override;

  // Custom allocator for this expression type.
  static boost::shared_ptr<ArrayLiteralExpression> Alloc(
      const Annotations &p_annotations, const TypeImpl &p_annotatedType,
      const std::vector<const Expression *> &p_children, VariableID p_id,
      TypeManager &p_typeManager);

  // Custom allocator for a flat array.
  static boost::shared_ptr<ArrayLiteralExpression> Alloc(
      const Annotations &p_annotations, const ArrayType &p_type,
      const std::vector<const Expression *> &p_children, VariableID p_id);

  // Iterate over children.
  const Expression *const *Begin() const;
  const Expression *const *End() const;

  // Flatten array, throwing an exception for non-square arrays, and for
  // arrays that contain non-literal arrays. If an annotated type is
  // provided to the latter function, the annotated type will be unified
  // with the child type of the array. In this case, both p_annotatedType
  // and p_typeManager must be non-NULL.
  const ArrayLiteralExpression &Flatten(SimpleExpressionOwner &p_owner) const;
  const ArrayLiteralExpression &Flatten(SimpleExpressionOwner &p_owner,
                                        const TypeImpl *p_annotatedType,
                                        TypeManager *p_typeManager) const;

  // Return a flag indicating whether or not this array has been
  // flattened. If this is true, all children of this expression are of
  // non-array type.
  bool IsFlat() const;

  // Gets the integer identificator for this array literal.
  VariableID GetId() const;

 private:
  // Construct an array literal with an annotated type.
  ArrayLiteralExpression(const Annotations &p_annotations,
                         const TypeImpl &p_annotatedType,
                         const std::vector<const Expression *> &p_children,
                         VariableID p_id, TypeManager &p_typeManager);

  // Construct a flat array.
  ArrayLiteralExpression(const Annotations &p_annotations,
                         const ArrayType &p_type,
                         const std::vector<const Expression *> &p_elements,
                         VariableID p_id);

  // Unify child types into the final array type.
  const ArrayType &UnifyTypes(const TypeImpl &p_annotatedType,
                              TypeManager &p_typeManager);

  // The type of this array literal.
  const ArrayType *m_type;

  // Custom deallocator for this expression type, suitable for
  // use as a shared_ptr destructor.
  static void DeleteAlloc(ArrayLiteralExpression *p_allocated);

  // Whether this array has been flattened.
  bool m_isFlat;

  // Number of children of this array.
  unsigned int m_numChildren;

  // An unique integer id for the current expression.
  const VariableID m_id;

  // Array of children of this node, allocated using struct hack.
  const Expression *m_children[1];
};
};  // namespace FreeForm2

#endif
