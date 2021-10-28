/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#pragma once

#ifndef FREEFORM2_LETEXPRESSION_H
#define FREEFORM2_LETEXPRESSION_H

#include <utility>
#include <vector>

#include "Expression.h"

namespace FreeForm2 {
class LetExpression : public Expression {
 public:
  typedef std::pair<VariableID, const Expression *> IdExpressionPair;

  // Methods inherited from Expression.
  virtual size_t GetNumChildren() const override;
  virtual void Accept(Visitor &p_visitor) const override;
  virtual const TypeImpl &GetType() const override;

  // Methods needed for the alternative visitation method.
  const Expression &GetValue() const;
  const IdExpressionPair *GetBound() const;

  // Custom allocation method.
  static boost::shared_ptr<LetExpression> Alloc(
      const Annotations &p_annotations,
      const std::vector<IdExpressionPair> &p_children,
      const Expression *p_value);

 private:
  // Private constructor for struct hack allocation.
  LetExpression(const Annotations &p_annotations,
                const std::vector<IdExpressionPair> &p_children,
                const Expression *p_value);

  // Custom deallocation method.
  static void DeleteAlloc(LetExpression *p_allocated);

  // Sub-expression that dictates the value of the let expression.
  const Expression *m_value;

  // Number of quantities (variables) bound by this let.
  unsigned int m_numBound;

  // Array of quantites, allocated via struct hack, bound by this let.
  IdExpressionPair m_bound[1];
};
}  // namespace FreeForm2

#endif
