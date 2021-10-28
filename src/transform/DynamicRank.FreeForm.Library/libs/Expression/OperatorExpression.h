/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#pragma once

#ifndef FREEFORM2_OPERATOR_EXPRESSION_H
#define FREEFORM2_OPERATOR_EXPRESSION_H

#include <vector>

#include "BinaryOperator.h"
#include "Expression.h"
#include "FreeForm2Type.h"
#include "UnaryOperator.h"

namespace FreeForm2 {
class BinaryOperator;
class TypeManager;

class UnaryOperatorExpression : public Expression {
 public:
  UnaryOperatorExpression(const Annotations &p_annotations,
                          const Expression &p_child,
                          UnaryOperator::Operation p_op,
                          ValueBounds p_valueBounds);
  UnaryOperatorExpression(const Annotations &p_annotations,
                          const Expression &p_child,
                          UnaryOperator::Operation p_op);

  virtual void Accept(Visitor &p_visitor) const override;
  virtual size_t GetNumChildren() const override;
  virtual const TypeImpl &GetType() const override;

  // Unary operator used in this expression.
  const UnaryOperator::Operation m_op;

  // Child of this node.
  const Expression &m_child;

 private:
  // Infer the resulting type of the operator expression.
  const TypeImpl &InferType() const;

  // The stored result type of the operator expression.
  const TypeImpl &m_type;
};

class BinaryOperatorExpression : public Expression {
 public:
  void Accept(Visitor &p_visitor) const override;
  virtual size_t GetNumChildren() const override;
  virtual const TypeImpl &GetType() const override;

  const TypeImpl &GetChildType() const;
  const Expression *const *GetChildren() const;

  BinaryOperator::Operation GetOperator() const;

  static boost::shared_ptr<BinaryOperatorExpression> Alloc(
      const Annotations &p_annotations,
      const std::vector<const Expression *> &p_children,
      const BinaryOperator::Operation p_binaryOp, TypeManager &p_typeManager);

  static boost::shared_ptr<BinaryOperatorExpression> Alloc(
      const Annotations &p_annotations,
      const std::vector<const Expression *> &p_children,
      const BinaryOperator::Operation p_binaryOp, TypeManager &p_typeManager,
      ValueBounds p_valueBounds);

  static boost::shared_ptr<BinaryOperatorExpression> Alloc(
      const Annotations &p_annotations, const Expression &p_leftChild,
      const Expression &p_rightChild,
      const BinaryOperator::Operation p_binaryOp, TypeManager &p_typeManager);

 private:
  // Constructors are private, call Alloc instead.
  BinaryOperatorExpression(const Annotations &p_annotations,
                           const std::vector<const Expression *> &p_children,
                           const BinaryOperator::Operation p_binaryOp,
                           TypeManager &p_typeManager);
  BinaryOperatorExpression(const Annotations &p_annotations,
                           const std::vector<const Expression *> &p_children,
                           const BinaryOperator::Operation p_binaryOp,
                           TypeManager &p_typeManager,
                           ValueBounds p_valueBounds);
  BinaryOperatorExpression(const Annotations &p_annotations,
                           const Expression &p_leftChild,
                           const Expression &p_rightChild,
                           const BinaryOperator::Operation p_binaryOp,
                           TypeManager &p_typeManager);

  // Infer the resulting type of this operator expression.
  const TypeImpl *m_resultType;
  const TypeImpl *m_childType;

  // Infer the type that children of this expression need to be.
  const TypeImpl &InferChildType(TypeManager &p_typeManager) const;

  // Binary operator used to compile arithmetic.
  const BinaryOperator::Operation m_binaryOp;

  // Number of children of this node.
  size_t m_numChildren;

  // The statically calculated bounds of the values this expression
  // can take.
  ValueBounds m_valueBounds;

  // Array of children of this node, allocated using struct hack.
  const Expression *m_children[1];

  static void DeleteAlloc(BinaryOperatorExpression *p_allocated);
};
}  // namespace FreeForm2

#endif
