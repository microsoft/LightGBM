/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include "StreamData.h"

#include "FreeForm2Assert.h"
#include "Visitor.h"

FreeForm2::StreamDataExpression::StreamDataExpression(
    const FreeForm2::Annotations &p_annotations, bool p_requestsLength)
    : Expression(p_annotations), m_requestsLength(p_requestsLength) {}

void FreeForm2::StreamDataExpression::Accept(Visitor &p_visitor) const {
  size_t stackSize = p_visitor.StackSize();

  if (!p_visitor.AlternativeVisit(*this)) {
    p_visitor.Visit(*this);
  }

  FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}

const FreeForm2::TypeImpl &FreeForm2::StreamDataExpression::GetType() const {
  return TypeImpl::GetIntInstance(true);
}

size_t FreeForm2::StreamDataExpression::GetNumChildren() const { return 0; }

const FreeForm2::UpdateStreamDataExpression &
FreeForm2::UpdateStreamDataExpression::GetInstance() {
  static const Annotations emptyAnnotations;
  static const UpdateStreamDataExpression exp(emptyAnnotations);
  return exp;
}

void FreeForm2::UpdateStreamDataExpression::Accept(Visitor &p_visitor) const {
  size_t stackSize = p_visitor.StackSize();

  if (!p_visitor.AlternativeVisit(*this)) {
    p_visitor.Visit(*this);
  }

  FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}

const FreeForm2::TypeImpl &FreeForm2::UpdateStreamDataExpression::GetType()
    const {
  return TypeImpl::GetVoidInstance();
}

size_t FreeForm2::UpdateStreamDataExpression::GetNumChildren() const {
  return 0;
}

FreeForm2::UpdateStreamDataExpression::UpdateStreamDataExpression(
    const FreeForm2::Annotations &p_annotations)
    : Expression(p_annotations) {}
