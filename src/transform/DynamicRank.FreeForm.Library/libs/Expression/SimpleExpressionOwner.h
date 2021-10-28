/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#pragma once

#ifndef FREEFORM2_SIMPLEEXPRESSIONOWNER
#define FREEFORM2_SIMPLEEXPRESSIONOWNER

#include <vector>

#include "Expression.h"

namespace FreeForm2 {
// Straight-forward expression owner, that keeps shared pointers
// to given expressions.
class SimpleExpressionOwner : public ExpressionOwner {
 public:
  // Transfer ownership of the given expression to the expression owner.
  const Expression *AddExpression(
      const boost::shared_ptr<const Expression> &p_expr) {
    m_exp.push_back(p_expr);
    return m_exp.back().get();
  }

 private:
  // Vector of managed expressions.
  std::vector<boost::shared_ptr<const Expression> > m_exp;
};
}  // namespace FreeForm2

#endif
