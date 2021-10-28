/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include "RandExpression.h"

#include "FreeForm2Assert.h"
#include "Visitor.h"

FreeForm2::RandFloatExpression::RandFloatExpression(
    const Annotations &p_annotations)
    : Expression(p_annotations) {}

void FreeForm2::RandFloatExpression::Accept(Visitor &p_visitor) const {
  const size_t stackSize = p_visitor.StackSize();

  if (!p_visitor.AlternativeVisit(*this)) {
    p_visitor.Visit(*this);
  }

  FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}

const FreeForm2::TypeImpl &FreeForm2::RandFloatExpression::GetType() const {
  return TypeImpl::GetFloatInstance(true);
}

size_t FreeForm2::RandFloatExpression::GetNumChildren() const { return 0; }

const FreeForm2::RandFloatExpression &
FreeForm2::RandFloatExpression::GetInstance() {
  static const Annotations s_annotations;
  static const RandFloatExpression s_instance(s_annotations);
  return s_instance;
}

FreeForm2::RandIntExpression::RandIntExpression(
    const Annotations &p_annotations, const Expression &p_lowerBoundExpression,
    const Expression &p_upperBoundExpression)
    : Expression(p_annotations),
      m_lowerBoundExpression(p_lowerBoundExpression),
      m_upperBoundExpression(p_upperBoundExpression) {}

void FreeForm2::RandIntExpression::Accept(Visitor &p_visitor) const {
  const size_t stackSize = p_visitor.StackSize();

  if (!p_visitor.AlternativeVisit(*this)) {
    m_lowerBoundExpression.Accept(p_visitor);
    m_upperBoundExpression.Accept(p_visitor);
    p_visitor.Visit(*this);
  }

  FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}

const FreeForm2::TypeImpl &FreeForm2::RandIntExpression::GetType() const {
  return TypeImpl::GetIntInstance(true);
}

size_t FreeForm2::RandIntExpression::GetNumChildren() const { return 2; }
