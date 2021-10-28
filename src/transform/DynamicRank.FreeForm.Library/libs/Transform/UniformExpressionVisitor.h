/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#pragma once

#ifndef FREEFORM2_UNIFORM_EXPRESSION_VISITOR_H
#define FREEFORM2_UNIFORM_EXPRESSION_VISITOR_H

#include "Visitor.h"

namespace FreeForm2 {
// Visitor that implements methods for every expression class that does nothing.
class UniformExpressionVisitor : public Visitor {
 public:
  virtual void Visit(const Expression &p_expr) = 0;

  // Methods inherited from Visitor.
  virtual void Visit(const SelectNthExpression &p_expr) override;
  virtual void Visit(const SelectRangeExpression &p_expr) override;
  virtual void Visit(const ConditionalExpression &p_expr) override;
  virtual void Visit(const ArrayLiteralExpression &p_expr) override;
  virtual void Visit(const LetExpression &p_expr) override;
  virtual void Visit(const BlockExpression &p_expr) override;
  virtual void Visit(const BinaryOperatorExpression &p_expr) override;
  virtual void Visit(const RangeReduceExpression &p_expr) override;
  virtual void Visit(const ForEachLoopExpression &p_expr) override;
  virtual void Visit(const ComplexRangeLoopExpression &p_expr) override;
  virtual void Visit(const MutationExpression &p_expr) override;
  virtual void Visit(const MatchExpression &p_expr) override;
  virtual void Visit(const MatchOperatorExpression &p_expr) override;
  virtual void Visit(const MatchGuardExpression &p_expr) override;
  virtual void Visit(const MatchBindExpression &p_expr) override;
  virtual void Visit(const MemberAccessExpression &p_expr) override;
  virtual void Visit(const ArrayLengthExpression &p_expr) override;
  virtual void Visit(const ArrayDereferenceExpression &p_expr) override;
  virtual void Visit(const ConvertToFloatExpression &p_expr) override;
  virtual void Visit(const ConvertToIntExpression &p_expr) override;
  virtual void Visit(const ConvertToUInt64Expression &p_expr) override;
  virtual void Visit(const ConvertToInt32Expression &p_expr) override;
  virtual void Visit(const ConvertToUInt32Expression &p_expr) override;
  virtual void Visit(const ConvertToBoolExpression &p_expr) override;
  virtual void Visit(const ConvertToImperativeExpression &p_expr) override;
  virtual void Visit(const DeclarationExpression &p_expr) override;
  virtual void Visit(const DirectPublishExpression &p_expr) override;
  virtual void Visit(const ExternExpression &p_expr) override;
  virtual void Visit(const FunctionExpression &p_expr) override;
  virtual void Visit(const FunctionCallExpression &p_expr) override;
  virtual void Visit(const LiteralIntExpression &p_expr) override;
  virtual void Visit(const LiteralUInt64Expression &p_expr) override;
  virtual void Visit(const LiteralInt32Expression &p_expr) override;
  virtual void Visit(const LiteralUInt32Expression &p_expr) override;
  virtual void Visit(const LiteralFloatExpression &p_expr) override;
  virtual void Visit(const LiteralBoolExpression &p_expr) override;
  virtual void Visit(const LiteralVoidExpression &p_expr) override;
  virtual void Visit(const LiteralStreamExpression &p_expr) override;
  virtual void Visit(const LiteralWordExpression &p_expr) override;
  virtual void Visit(const LiteralInstanceHeaderExpression &p_expr) override;
  virtual void Visit(const FeatureRefExpression &p_expr) override;
  virtual void Visit(const UnaryOperatorExpression &p_expr) override;
  virtual void Visit(const FeatureSpecExpression &p_expr) override;
  virtual void Visit(const FeatureGroupSpecExpression &p_expr) override;
  virtual void Visit(const PhiNodeExpression &p_expr) override;
  virtual void Visit(const PublishExpression &p_expr) override;
  virtual void Visit(const ReturnExpression &p_expr) override;
  virtual void Visit(const StreamDataExpression &p_expr) override;
  virtual void Visit(const UpdateStreamDataExpression &p_expr) override;
  virtual void Visit(const VariableRefExpression &p_expr) override;
  virtual void Visit(const ImportFeatureExpression &p_expr) override;
  virtual void Visit(const StateExpression &p_expr) override;
  virtual void Visit(const StateMachineExpression &p_expr) override;
  virtual void Visit(
      const ExecuteStreamRewritingStateMachineGroupExpression &p_expr) override;
  virtual void Visit(const ExecuteMachineExpression &p_expr) override;
  virtual void Visit(const ExecuteMachineGroupExpression &p_expr) override;
  virtual void Visit(const YieldExpression &p_expr) override;
  virtual void Visit(const RandFloatExpression &p_expr) override;
  virtual void Visit(const RandIntExpression &p_expr) override;
  virtual void Visit(const ThisExpression &p_expr) override;
  virtual void Visit(const UnresolvedAccessExpression &p_expr) override;
  virtual void Visit(const TypeInitializerExpression &p_expr) override;
  virtual void Visit(const AggregateContextExpression &p_expr) override;
  virtual void Visit(const DebugExpression &p_expr) override;

  virtual void VisitReference(
      const ArrayDereferenceExpression &p_expr) override;
  virtual void VisitReference(const VariableRefExpression &p_expr) override;
  virtual void VisitReference(const MemberAccessExpression &p_expr) override;
  virtual void VisitReference(const ThisExpression &) override;
  virtual void VisitReference(const UnresolvedAccessExpression &) override;
};
}  // namespace FreeForm2

#endif
