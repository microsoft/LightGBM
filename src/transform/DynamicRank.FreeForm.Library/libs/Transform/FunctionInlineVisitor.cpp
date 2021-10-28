/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include "FunctionInlineVisitor.h"

#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include <sstream>

#include "FreeForm2Assert.h"
#include "Function.h"
#include "FunctionType.h"
#include "LetExpression.h"
#include "RefExpression.h"
#include "TypeImpl.h"
#include "TypeUtil.h"

FreeForm2::FunctionInlineVisitor::FunctionInlineVisitor(
    const boost::shared_ptr<SimpleExpressionOwner> &p_owner,
    const boost::shared_ptr<TypeManager> &p_typeManager,
    VariableID p_variableId)
    : CopyingVisitor(p_owner, p_typeManager), m_variableId(p_variableId) {}

FreeForm2::VariableID FreeForm2::FunctionInlineVisitor::GetVariableId() {
  return m_variableId;
}

bool FreeForm2::FunctionInlineVisitor::AlternativeVisit(
    const FunctionCallExpression &p_expr) {
  const FunctionType &funType = p_expr.GetFunctionType();
  FF2_ASSERT(funType.GetParameterCount() == p_expr.GetNumParameters());

  std::vector<LetExpression::IdExpressionPair> letValues;
  letValues.reserve(funType.GetParameterCount());

  std::vector<const Expression *> parameterExpressions;
  parameterExpressions.reserve(funType.GetParameterCount());

  const FunctionExpression &functionExpression =
      dynamic_cast<const FunctionExpression &>(p_expr.GetFunction());
  const std::vector<FunctionExpression::Parameter> &functionParams =
      functionExpression.GetParameters();
  FF2_ASSERT(functionParams.size() == funType.GetParameterCount());

  for (size_t i = 0; i < funType.GetParameterCount(); i++) {
    // Visit the parameter to make sure that its type has been determined.
    // (It could be a FunctionCallExpression for instance)
    p_expr.GetParameters()[i]->Accept(*this);
    const Expression *parameter = m_stack.back();
    m_stack.pop_back();

    const TypeImpl *formalType = funType.BeginParameters()[i];
    FF2_ASSERT(formalType != nullptr && parameter != nullptr);
    const TypeImpl &paramType = parameter->GetType();
    FF2_ASSERT(paramType.Primitive() != Type::Unknown);

    // Try to assign the type of the parameter to the type of the Function
    // input.
    FF2_ASSERT(TypeUtil::IsAssignable(*formalType, paramType));

    if (formalType->Primitive() == Type::Unknown) {
      formalType = &paramType;
    } else if (*formalType != paramType) {
      FF2_ASSERT(TypeUtil::IsConvertible(paramType, *formalType));
      auto expr = TypeUtil::Convert(*parameter, formalType->Primitive());
      AddExpressionToOwner(expr);
      parameter = expr.get();
    }
    m_parameterTypeTranslation.insert(
        std::make_pair(functionParams[i].m_parameter->GetId(), &paramType));
    parameterExpressions.push_back(parameter);
  }

  // Determine new variable ids.
  FF2_ASSERT(m_newVariableIdMapping.empty());
  for (size_t i = 0; i < funType.GetParameterCount(); i++) {
    VariableID newVariableID = m_variableId;
    ++m_variableId.m_value;
    m_newVariableIdMapping.insert(
        std::make_pair(functionParams[i].m_parameter->GetId(), newVariableID));
    letValues.push_back(std::make_pair(newVariableID, parameterExpressions[i]));
  }

  // Then visit the Function Body. This should replace unknown types with known
  // types. And old variable ids with the new ids.
  functionExpression.GetBody().Accept(*this);
  const Expression *newFunctionBody = m_stack.back();
  m_stack.pop_back();

  // Clear out the variable id mapping.
  m_newVariableIdMapping.clear();

  // Add a Let expression containing the parameter values in the Invoke
  // statement and the function's body.
  AddExpression(LetExpression::Alloc(p_expr.GetAnnotations(), letValues,
                                     newFunctionBody));

  return true;
}

void FreeForm2::FunctionInlineVisitor::Visit(
    const VariableRefExpression &p_expr) {
  const auto find = m_parameterTypeTranslation.find(p_expr.GetId());
  if (find != m_parameterTypeTranslation.end()) {
    const auto newVariableIdFind = m_newVariableIdMapping.find(p_expr.GetId());
    FF2_ASSERT(newVariableIdFind != m_newVariableIdMapping.end());
    AddExpression(boost::make_shared<VariableRefExpression>(
        p_expr.GetAnnotations(), newVariableIdFind->second, p_expr.GetVersion(),
        *find->second));
  } else {
    CopyingVisitor::Visit(p_expr);
  }
}

void FreeForm2::FunctionInlineVisitor::Visit(const ReturnExpression &p_expr) {
  // The FunctionInlineVisitor should never process a ReturnExpression.
  FF2_UNREACHABLE();
}
