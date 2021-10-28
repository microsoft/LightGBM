/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include "LetExpression.h"

#include <sstream>

#include "FreeForm2Assert.h"
#include "RefExpression.h"
#include "SimpleExpressionOwner.h"
#include "Visitor.h"

FreeForm2::LetExpression::LetExpression(
    const Annotations &p_annotations,
    const std::vector<IdExpressionPair> &p_children, const Expression *p_value)
    : Expression(p_annotations),
      m_numBound(static_cast<unsigned int>(p_children.size())),
      m_value(p_value) {
  for (unsigned int i = 0; i < p_children.size(); i++) {
    m_bound[i] = p_children[i];
  }
}

const FreeForm2::TypeImpl &FreeForm2::LetExpression::GetType() const {
  return m_value->GetType();
}

size_t FreeForm2::LetExpression::GetNumChildren() const {
  return m_numBound + 1;
}

const FreeForm2::Expression &FreeForm2::LetExpression::GetValue() const {
  return *m_value;
}

const FreeForm2::LetExpression::IdExpressionPair *
FreeForm2::LetExpression::GetBound() const {
  return m_bound;
}

void FreeForm2::LetExpression::Accept(Visitor &p_visitor) const {
  size_t stackSize = p_visitor.StackSize();

  if (!p_visitor.AlternativeVisit(*this)) {
    for (unsigned int i = 0; i < m_numBound; i++) {
      m_bound[i].second->Accept(p_visitor);
    }

    m_value->Accept(p_visitor);
    p_visitor.Visit(*this);
  }

  FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}

boost::shared_ptr<FreeForm2::LetExpression> FreeForm2::LetExpression::Alloc(
    const Annotations &p_annotations,
    const std::vector<IdExpressionPair> &p_children,
    const Expression *p_value) {
  size_t bytes = sizeof(LetExpression) +
                 (p_children.size() - 1) * sizeof(IdExpressionPair);

  // Constructor assertions must appear in the allocation method: the
  // constructor may not throw; otherwise, the raw memory allocation will
  // leak.
  FF2_ASSERT(!p_children.empty());
  for (size_t i = 0; i < p_children.size(); i++) {
    for (size_t j = 0; j < i; j++) {
      FF2_ASSERT(p_children[j].first != p_children[i].first);
    }
  }

  // Allocate a shared_ptr that deletes an LetExpression
  // allocated in a char[].
  boost::shared_ptr<LetExpression> exp(
      new (new char[bytes]) LetExpression(p_annotations, p_children, p_value),
      DeleteAlloc);
  return exp;
}

void FreeForm2::LetExpression::DeleteAlloc(LetExpression *p_allocated) {
  // Manually call dtor for let expression.
  p_allocated->~LetExpression();

  // Dispose of memory, which we allocated in a char[].
  char *mem = reinterpret_cast<char *>(p_allocated);
  delete[] mem;
}
