/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include "Match.h"

#include <sstream>

#include "FreeForm2Assert.h"
#include "Visitor.h"

FreeForm2::MatchExpression::MatchExpression(const Annotations &p_annotations,
                                            const Expression &p_value,
                                            const MatchSubExpression &p_pattern,
                                            const Expression &p_action,
                                            bool p_isOverlapping)
    : Expression(p_annotations),
      m_value(p_value),
      m_pattern(p_pattern),
      m_action(p_action),
      m_isOverlapping(p_isOverlapping) {}

void FreeForm2::MatchExpression::Accept(Visitor &p_visitor) const {
  size_t stackSize = p_visitor.StackSize();

  if (!p_visitor.AlternativeVisit(*this)) {
    m_value.Accept(p_visitor);
    m_pattern.Accept(p_visitor);
    m_action.Accept(p_visitor);

    p_visitor.Visit(*this);
  }

  FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}

const FreeForm2::TypeImpl &FreeForm2::MatchExpression::GetType() const {
  return TypeImpl::GetVoidInstance();
}

size_t FreeForm2::MatchExpression::GetNumChildren() const { return 3; }

const FreeForm2::Expression &FreeForm2::MatchExpression::GetValue() const {
  return m_value;
}

const FreeForm2::MatchSubExpression &FreeForm2::MatchExpression::GetPattern()
    const {
  return m_pattern;
}

const FreeForm2::Expression &FreeForm2::MatchExpression::GetAction() const {
  return m_action;
}

bool FreeForm2::MatchExpression::IsOverlapping() const {
  return m_isOverlapping;
}

void FreeForm2::MatchOperatorExpression::Accept(Visitor &p_visitor) const {
  size_t stackSize = p_visitor.StackSize();

  if (!p_visitor.AlternativeVisit(*this)) {
    for (size_t i = 0; i < m_numChildren; i++) {
      m_children[i]->Accept(p_visitor);
    }

    p_visitor.Visit(*this);
  }

  FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}

const FreeForm2::TypeImpl &FreeForm2::MatchOperatorExpression::GetType() const {
  return TypeImpl::GetVoidInstance();
}

size_t FreeForm2::MatchOperatorExpression::GetNumChildren() const {
  return m_numChildren;
}

FreeForm2::MatchSubExpression::Info
FreeForm2::MatchOperatorExpression::GetInfo() const {
  switch (GetOperator()) {
    case kleene: {
      FF2_ASSERT(m_numChildren == 1);
      return Info(0, Info::c_indeterminate);
    }

    case atLeastOne: {
      FF2_ASSERT(m_numChildren == 1);
      return Info(m_children[0]->GetInfo().m_minLength, Info::c_indeterminate);
    }

    case alternation: {
      unsigned int min = UINT_MAX;
      unsigned int max = 0;

      for (size_t i = 0; i < GetNumChildren(); i++) {
        Info info = m_children[i]->GetInfo();
        min = std::min(min, info.m_minLength);
        max = std::max(max, info.m_maxLength);
      }

      return Info(min, max);
    }

    case concatenation: {
      Info combined(0, 0);

      for (size_t i = 0; i < GetNumChildren(); i++) {
        Info info = m_children[i]->GetInfo();
        combined.m_minLength += info.m_minLength;

        if (combined.m_maxLength != Info::c_indeterminate &&
            info.m_maxLength != Info::c_indeterminate) {
          combined.m_maxLength += info.m_maxLength;
        } else {
          combined.m_maxLength = Info::c_indeterminate;
        }
      }

      return combined;
    }

    default: {
      Unreachable(__FILE__, __LINE__);
      break;
    }
  }
}

boost::shared_ptr<FreeForm2::MatchOperatorExpression>
FreeForm2::MatchOperatorExpression::Alloc(const Annotations &p_annotations,
                                          const MatchSubExpression **p_children,
                                          size_t p_numChildren, Operator p_op) {
  size_t bytes = sizeof(MatchOperatorExpression) +
                 sizeof(Expression *) * (p_numChildren - 1);

  // Allocate a shared_ptr that deletes an MatchOperatorExpression
  // allocated in a char[].
  boost::shared_ptr<MatchOperatorExpression> exp(
      new (new char[bytes]) MatchOperatorExpression(p_annotations, p_children,
                                                    p_numChildren, p_op),
      DeleteAlloc);
  return exp;
}

boost::shared_ptr<FreeForm2::MatchOperatorExpression>
FreeForm2::MatchOperatorExpression::Alloc(const Annotations &p_annotations,
                                          const MatchSubExpression &p_left,
                                          const MatchSubExpression &p_right,
                                          Operator p_op) {
  const MatchSubExpression *array[2];
  array[0] = &p_left;
  array[1] = &p_right;
  return Alloc(p_annotations, array, sizeof(array) / sizeof(*array), p_op);
}

boost::shared_ptr<FreeForm2::MatchOperatorExpression>
FreeForm2::MatchOperatorExpression::Alloc(const Annotations &p_annotations,
                                          const MatchSubExpression &p_expr,
                                          Operator p_op) {
  const MatchSubExpression *array[1];
  array[0] = &p_expr;
  return Alloc(p_annotations, array, sizeof(array) / sizeof(*array), p_op);
}

FreeForm2::MatchOperatorExpression::Operator
FreeForm2::MatchOperatorExpression::GetOperator() const {
  return m_op;
}

const FreeForm2::MatchSubExpression *const *
FreeForm2::MatchOperatorExpression::GetChildren() const {
  return m_children;
}

FreeForm2::MatchOperatorExpression::MatchOperatorExpression(
    const Annotations &p_annotations, const MatchSubExpression **p_children,
    size_t p_numChildren, Operator p_op)
    : MatchSubExpression(p_annotations),
      m_numChildren(p_numChildren),
      m_op(p_op) {
  for (size_t i = 0; i < p_numChildren; i++) {
    m_children[i] = p_children[i];
  }
}

void FreeForm2::MatchOperatorExpression::DeleteAlloc(
    MatchOperatorExpression *p_allocated) {
  // Manually call dtor for operator expression.
  p_allocated->~MatchOperatorExpression();

  // Dispose of memory, which we allocated in a char[].
  char *mem = reinterpret_cast<char *>(p_allocated);
  delete[] mem;
}

FreeForm2::MatchGuardExpression::MatchGuardExpression(
    const Annotations &p_annotations, const Expression &p_guard)
    : MatchSubExpression(p_annotations), m_guard(p_guard) {}

void FreeForm2::MatchGuardExpression::Accept(Visitor &p_visitor) const {
  size_t stackSize = p_visitor.StackSize();

  if (!p_visitor.AlternativeVisit(*this)) {
    m_guard.Accept(p_visitor);
    p_visitor.Visit(*this);
  }

  FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}

const FreeForm2::TypeImpl &FreeForm2::MatchGuardExpression::GetType() const {
  if (m_guard.GetType().Primitive() != Type::Bool) {
    std::ostringstream err;
    err << "Guard expression must evaluate to a boolean (evalutes to "
        << m_guard.GetType() << ")";
    throw ParseError(err.str(), GetSourceLocation());
  }

  return TypeImpl::GetBoolInstance(true);
}

size_t FreeForm2::MatchGuardExpression::GetNumChildren() const { return 1; }

FreeForm2::MatchSubExpression::Info FreeForm2::MatchGuardExpression::GetInfo()
    const {
  return Info(0, 0);
}

FreeForm2::MatchBindExpression::MatchBindExpression(
    const Annotations &p_annotations, const MatchSubExpression &p_value,
    VariableID p_id)
    : MatchSubExpression(p_annotations), m_value(p_value), m_id(p_id) {}

void FreeForm2::MatchBindExpression::Accept(Visitor &p_visitor) const {
  size_t stackSize = p_visitor.StackSize();

  if (!p_visitor.AlternativeVisit(*this)) {
    m_value.Accept(p_visitor);

    p_visitor.Visit(*this);
  }

  FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}

const FreeForm2::TypeImpl &FreeForm2::MatchBindExpression::GetType() const {
  return m_value.GetType();
}

size_t FreeForm2::MatchBindExpression::GetNumChildren() const { return 1; }

FreeForm2::MatchSubExpression::Info FreeForm2::MatchBindExpression::GetInfo()
    const {
  return m_value.GetInfo();
}
