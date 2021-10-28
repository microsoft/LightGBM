/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#pragma once

#include <map>

#include "CopyingVisitor.h"

namespace FreeForm2 {
// This class inlines function calls, inserting the function body directly
// into the resultant expression tree. This class only works with
// S-expression trees (or trees which have an implicit function return).
class FunctionInlineVisitor : public CopyingVisitor {
 public:
  FunctionInlineVisitor(const boost::shared_ptr<SimpleExpressionOwner> &p_owner,
                        const boost::shared_ptr<TypeManager> &p_typeManager,
                        VariableID p_variableId);

  virtual bool AlternativeVisit(const FunctionCallExpression &p_expr) override;
  virtual void Visit(const ReturnExpression &p_expr) override;
  virtual void Visit(const VariableRefExpression &p_expr) override;

  // Returns the variable id counter.
  VariableID GetVariableId();

 private:
  // A map containing type translations for function parameters.
  std::map<VariableID, const TypeImpl *> m_parameterTypeTranslation;

  // A map containing mappings from old variable ids to new variable ids.
  // This is necessary for all variables within a lambda.
  std::map<VariableID, VariableID> m_newVariableIdMapping;

  // Counter to keep track of the next variable ID.  This allows us
  // to ensure that the variables assigned for each function call have
  // unique ids.
  VariableID m_variableId;
};
}  // namespace FreeForm2