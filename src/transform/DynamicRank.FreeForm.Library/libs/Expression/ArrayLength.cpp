/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include "ArrayLength.h"

#include <sstream>

#include "Expression.h"
#include "FreeForm2Assert.h"
#include "SimpleExpressionOwner.h"
#include "Visitor.h"

FreeForm2::ArrayLengthExpression::ArrayLengthExpression(
    const Annotations &p_annotations, const Expression &p_array)
    : Expression(p_annotations), m_array(p_array) {}

void FreeForm2::ArrayLengthExpression::Accept(Visitor &p_visitor) const {
  size_t stackSize = p_visitor.StackSize();

  if (!p_visitor.AlternativeVisit(*this)) {
    m_array.Accept(p_visitor);

    p_visitor.Visit(*this);
  }

  FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}

const FreeForm2::TypeImpl &FreeForm2::ArrayLengthExpression::GetType() const {
  if (m_array.GetType().Primitive() != Type::Array) {
    std::ostringstream err;
    err << "Argument to array-length expression must be "
        << "an array (got type '" << m_array.GetType() << "')";
    throw ParseError(err.str(), GetSourceLocation());
  }

  return TypeImpl::GetUInt32Instance(true);
}

size_t FreeForm2::ArrayLengthExpression::GetNumChildren() const { return 1; }
