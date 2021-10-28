/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include "RefExpression.h"

#include <INeuralNetFeatures.h>

#include <boost/static_assert.hpp>
#include <sstream>

#include "CompoundType.h"
#include "ConvertExpression.h"
#include "Declaration.h"
#include "FreeForm2Assert.h"
#include "TypeUtil.h"
#include "Visitor.h"

using namespace FreeForm2;
FreeForm2::FeatureRefExpression::FeatureRefExpression(
    const FreeForm2::Annotations &p_annotations, UInt32 p_index)
    : FreeForm2::Expression(p_annotations), m_index(p_index) {}

const FreeForm2::TypeImpl &FreeForm2::FeatureRefExpression::GetType() const {
  return TypeImpl::GetIntInstance(true);
}

size_t FreeForm2::FeatureRefExpression::GetNumChildren() const { return 0; }

void FreeForm2::FeatureRefExpression::Accept(Visitor &p_visitor) const {
  size_t stackSize = p_visitor.StackSize();

  if (!p_visitor.AlternativeVisit(*this)) {
    p_visitor.Visit(*this);
  }

  FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}

FreeForm2::VariableRefExpression::VariableRefExpression(
    const FreeForm2::Annotations &p_annotations, VariableID p_id,
    size_t p_version, const TypeImpl &p_type)
    : Expression(p_annotations),
      m_id(p_id),
      m_version(p_version),
      m_type(p_type) {}

FreeForm2::VariableRefExpression::~VariableRefExpression() {}

size_t FreeForm2::VariableRefExpression::GetNumChildren() const { return 0; }

void FreeForm2::VariableRefExpression::Accept(Visitor &p_visitor) const {
  size_t stackSize = p_visitor.StackSize();

  if (!p_visitor.AlternativeVisit(*this)) {
    p_visitor.Visit(*this);
  }

  FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}

void FreeForm2::VariableRefExpression::AcceptReference(
    Visitor &p_visitor) const {
  size_t stackSize = p_visitor.StackSize();
  p_visitor.VisitReference(*this);
  FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}

const FreeForm2::TypeImpl &FreeForm2::VariableRefExpression::GetType() const {
  return m_type;
}

FreeForm2::VariableID FreeForm2::VariableRefExpression::GetId() const {
  return m_id;
}

size_t FreeForm2::VariableRefExpression::GetVersion() const {
  return m_version;
}

FreeForm2::ThisExpression::ThisExpression(const Annotations &p_annotations,
                                          const TypeImpl &p_type)
    : Expression(p_annotations), m_type(p_type) {}

size_t FreeForm2::ThisExpression::GetNumChildren() const { return 0; }

void FreeForm2::ThisExpression::Accept(Visitor &p_visitor) const {
  const size_t stackSize = p_visitor.StackSize();
  p_visitor.Visit(*this);
  FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}

void FreeForm2::ThisExpression::AcceptReference(Visitor &p_visitor) const {
  const size_t stackSize = p_visitor.StackSize();
  p_visitor.VisitReference(*this);
  FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}

const FreeForm2::TypeImpl &FreeForm2::ThisExpression::GetType() const {
  FF2_ASSERT(m_type.Primitive() == Type::Unknown ||
             CompoundType::IsCompoundType(m_type));
  return m_type;
}
