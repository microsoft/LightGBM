/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include "OperandPromotionVisitor.h"

#include <boost/concept_check.hpp>
#include <boost/iterator/iterator_concepts.hpp>
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include <sstream>

#include "Conditional.h"
#include "ConvertExpression.h"
#include "Expression.h"
#include "FreeForm2Assert.h"
#include "Function.h"
#include "FunctionType.h"
#include "OperatorExpression.h"
#include "SimpleExpressionOwner.h"
#include "TypeUtil.h"

namespace {
// Unify the types of loop counter variables starting at an iterator
// position. The iterator must be a writable iterator.
template <typename Iter>
void UnifyLoopCounters(const Iter p_iter, unsigned int p_numCounters,
                       FreeForm2::TypeManager &p_typeManager,
                       FreeForm2::SimpleExpressionOwner &p_owner) {
  using namespace FreeForm2;
  BOOST_CONCEPT_ASSERT(
      (boost_concepts::WritableIterator<Iter, const Expression *>));
  const TypeImpl *unifiedType = &TypeImpl::GetUnknownType();

  Iter iter = p_iter;
  for (size_t i = 0; i < p_numCounters; ++i, ++iter) {
    unifiedType = &TypeUtil::Unify(*unifiedType, (*iter)->GetType(),
                                   p_typeManager, false, true);
  }

  if (!unifiedType->IsValid()) {
    std::ostringstream err;
    err << "Loop bounds must be of a unifiable type.";
    throw ParseError(err.str(), (*p_iter)->GetSourceLocation());
  }

  iter = p_iter;
  for (size_t i = 0; i < p_numCounters; ++i, ++iter) {
    const Expression &expression = **iter;

    if (!expression.GetType().IsSameAs(*unifiedType, true)) {
      const TypeImpl &stackType = expression.GetType();
      if (TypeUtil::IsConvertible(stackType, *unifiedType)) {
        boost::shared_ptr<Expression> convert(
            TypeUtil::Convert(expression, unifiedType->Primitive()));

        p_owner.AddExpression(convert);
        *iter = convert.get();
      } else {
        std::ostringstream err;
        err << "Expected a type convertible to " << *unifiedType
            << "got type: " << expression.GetType();
        throw ParseError(err.str(), expression.GetSourceLocation());
      }
    }
  }
}
}  // namespace

bool FreeForm2::OperandPromotionVisitor::AlternativeVisit(
    const BinaryOperatorExpression &p_expr) {
  // Handle via Visit: overridden to ensure Visit is called.
  return false;
}

void FreeForm2::OperandPromotionVisitor::Visit(
    const BinaryOperatorExpression &p_expr) {
  std::vector<const Expression *>::reverse_iterator iter = GetStack().rbegin();

  const TypeImpl &parameterType = p_expr.GetChildType().AsConstType();

  for (size_t i = 0; i < p_expr.GetNumChildren(); ++i, ++iter) {
    FF2_ASSERT(iter != GetStack().rend());
    const Expression &expression = **iter;
    if (!expression.GetType().IsSameAs(parameterType, true)) {
      const TypeImpl &stackType = expression.GetType();
      if (TypeUtil::IsConvertible(stackType, parameterType)) {
        boost::shared_ptr<Expression> convert(
            TypeUtil::Convert(expression, parameterType.Primitive()));

        AddExpressionToOwner(convert);
        *iter = convert.get();
      } else {
        std::ostringstream err;
        err << "Expected a type convertible to " << parameterType
            << "got type: " << expression.GetType();
        throw ParseError(err.str(), p_expr.GetSourceLocation());
      }
    }
  }

  CopyingVisitor::Visit(p_expr);
}

void FreeForm2::OperandPromotionVisitor::Visit(
    const FunctionCallExpression &p_expr) {
  auto iter = GetStack().rbegin();

  const FunctionType &type = p_expr.GetFunctionType();
  FunctionType::ParameterIterator parameterTypes = type.BeginParameters();

  for (size_t i = 0; i < p_expr.GetNumParameters(); ++i, ++iter) {
    FF2_ASSERT(iter != GetStack().rend());
    const Expression &expression = **iter;
    const TypeImpl &parameterType =
        *parameterTypes[p_expr.GetNumParameters() - i - 1];

    if (!expression.GetType().IsSameAs(parameterType, true)) {
      const TypeImpl &stackType = expression.GetType();
      if (TypeUtil::IsConvertible(stackType, parameterType)) {
        boost::shared_ptr<Expression> convert(
            TypeUtil::Convert(expression, parameterType.Primitive()));

        AddExpressionToOwner(convert);
        *iter = convert.get();
      } else {
        std::ostringstream err;
        err << "Expected a type convertible to " << parameterType
            << "got type: " << expression.GetType();
        throw ParseError(err.str(), p_expr.GetSourceLocation());
      }
    }
  }

  CopyingVisitor::Visit(p_expr);
}

void FreeForm2::OperandPromotionVisitor::Visit(
    const ConditionalExpression &p_expr) {
  FF2_ASSERT(GetStack().size() >= 3);

  auto iter = GetStack().rbegin() + 1;  // Skip condition expression.
  const TypeImpl &unifiedType = p_expr.GetType();

  for (size_t i = 0; i < 2; ++i, ++iter) {
    FF2_ASSERT(iter != GetStack().rend());
    const Expression &expression = **iter;

    if (!expression.GetType().IsSameAs(unifiedType, true)) {
      const TypeImpl &stackType = expression.GetType();
      if (TypeUtil::IsConvertible(stackType, unifiedType)) {
        boost::shared_ptr<Expression> convert(
            TypeUtil::Convert(expression, unifiedType.Primitive()));

        AddExpressionToOwner(convert);
        *iter = convert.get();
      } else {
        std::ostringstream err;
        err << "Expected a type convertible to " << unifiedType
            << "got type: " << expression.GetType();
        throw ParseError(err.str(), p_expr.GetSourceLocation());
      }
    }
  }

  CopyingVisitor::Visit(p_expr);
}

void FreeForm2::OperandPromotionVisitor::Visit(
    const RangeReduceExpression &p_expr) {
  FF2_ASSERT(GetStack().size() >= 4);

  UnifyLoopCounters(GetStack().rbegin() + 1, 2, *m_typeManager, *m_owner);

  CopyingVisitor::Visit(p_expr);
}

void FreeForm2::OperandPromotionVisitor::Visit(
    const ForEachLoopExpression &p_expr) {
  FF2_ASSERT(GetStack().size() >= 4);

  UnifyLoopCounters(GetStack().rbegin() + 1, 3, *m_typeManager, *m_owner);

  CopyingVisitor::Visit(p_expr);
}

void FreeForm2::OperandPromotionVisitor::Visit(
    const ComplexRangeLoopExpression &p_expr) {
  FF2_ASSERT(GetStack().size() >= 6);

  UnifyLoopCounters(GetStack().rbegin() + 2, 3, *m_typeManager, *m_owner);

  CopyingVisitor::Visit(p_expr);
}
