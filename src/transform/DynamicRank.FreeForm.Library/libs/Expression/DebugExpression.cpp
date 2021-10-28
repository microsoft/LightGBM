/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include "DebugExpression.h"

#include <sstream>

#include "ArrayType.h"
#include "FreeForm2Assert.h"
#include "TypeImpl.h"
#include "Visitor.h"

using namespace FreeForm2;

DebugExpression::DebugExpression(const Annotations &p_annotations,
                                 const Expression &p_child,
                                 const std::string &p_childText)
    : Expression(p_annotations), m_child(p_child), m_childText(p_childText) {}

void DebugExpression::Accept(Visitor &p_visitor) const {
  const size_t stackSize = p_visitor.StackSize();

  if (!p_visitor.AlternativeVisit(*this)) {
    m_child.Accept(p_visitor);
    p_visitor.Visit(*this);
  }

  FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}

const TypeImpl &DebugExpression::GetType() const {
  const TypeImpl &childType = m_child.GetType();
  const TypeImpl *checkType = &childType;
  if (childType.Primitive() == Type::Array) {
    const ArrayType &type = static_cast<const ArrayType &>(childType);
    checkType = &type.GetChildType();
  }

  if (!checkType->IsLeafType()) {
    std::ostringstream err;
    err << "Cannot debug the expression " << m_childText << " of type "
        << *checkType << ". Only arrays and primitive types are supported.";
    throw std::runtime_error(err.str());
  }

  return TypeImpl::GetVoidInstance();
}

size_t DebugExpression::GetNumChildren() const { return 1; }

const Expression &DebugExpression::GetChild() const { return m_child; }

const std::string &DebugExpression::GetChildText() const { return m_childText; }
