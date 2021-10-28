/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#pragma once

#ifndef FREEFORM2_NOOP_VISITOR_H
#define FREEFORM2_NOOP_VISITOR_H

#include "Visitor.h"

namespace FreeForm2 {
// Visitor that implements methods for every expression class that do nothing.
class NoOpVisitor : public Visitor {
 public:
  // Method inherited from Visitor.
  virtual void Allocate(const Allocation &) {}

  virtual void Visit(const SelectNthExpression &) override {}
  virtual void Visit(const SelectRangeExpression &) override {}
  virtual void Visit(const ConditionalExpression &) override {}
  virtual void Visit(const ArrayLiteralExpression &) override {}
  virtual void Visit(const LetExpression &) override {}
  virtual void Visit(const BlockExpression &) override {}
  virtual void Visit(const BinaryOperatorExpression &) override {}
  virtual void Visit(const RangeReduceExpression &) override {}
  virtual void Visit(const ForEachLoopExpression &) override {}
  virtual void Visit(const ComplexRangeLoopExpression &) override {}
  virtual void Visit(const MutationExpression &) override {}
  virtual void Visit(const MatchExpression &) override {}
  virtual void Visit(const MatchOperatorExpression &) override {}
  virtual void Visit(const MatchGuardExpression &) override {}
  virtual void Visit(const MatchBindExpression &) override {}
  virtual void Visit(const MemberAccessExpression &) override {}
  virtual void Visit(const PhiNodeExpression &) override {}
  virtual void Visit(const PublishExpression &) override {}
  virtual void Visit(const ReturnExpression &) override {}
  virtual void Visit(const ArrayLengthExpression &) override {}
  virtual void Visit(const ArrayDereferenceExpression &) override {}
  virtual void Visit(const ConvertToFloatExpression &) override {}
  virtual void Visit(const ConvertToIntExpression &) override {}
  virtual void Visit(const ConvertToUInt64Expression &) override {}
  virtual void Visit(const ConvertToInt32Expression &) override {}
  virtual void Visit(const ConvertToUInt32Expression &) override {}
  virtual void Visit(const ConvertToBoolExpression &) override {}
  virtual void Visit(const ConvertToImperativeExpression &) override {}
  virtual void Visit(const DeclarationExpression &) override {}
  virtual void Visit(const DirectPublishExpression &) override {}
  virtual void Visit(const ExternExpression &) override {}
  virtual void Visit(const FunctionExpression &) override {}
  virtual void Visit(const FunctionCallExpression &) override {}
  virtual void Visit(const LiteralIntExpression &) override {}
  virtual void Visit(const LiteralUInt64Expression &) override {}
  virtual void Visit(const LiteralInt32Expression &) override {}
  virtual void Visit(const LiteralUInt32Expression &) override {}
  virtual void Visit(const LiteralFloatExpression &) override {}
  virtual void Visit(const LiteralBoolExpression &) override {}
  virtual void Visit(const LiteralVoidExpression &) override {}
  virtual void Visit(const LiteralStreamExpression &) override {}
  virtual void Visit(const LiteralWordExpression &) override {}
  virtual void Visit(const LiteralInstanceHeaderExpression &) override {}
  virtual void Visit(const FeatureRefExpression &) override {}
  virtual void Visit(const UnaryOperatorExpression &) override {}
  virtual void Visit(const FeatureSpecExpression &) override {}
  virtual void Visit(const FeatureGroupSpecExpression &) override {}
  virtual void Visit(const StreamDataExpression &) override {}
  virtual void Visit(const UpdateStreamDataExpression &) override {}
  virtual void Visit(const VariableRefExpression &) override {}
  virtual void Visit(const ImportFeatureExpression &) override {}
  virtual void Visit(const StateExpression &) override {}
  virtual void Visit(const StateMachineExpression &) override {}
  virtual void Visit(
      const ExecuteStreamRewritingStateMachineGroupExpression &) override {}
  virtual void Visit(const ExecuteMachineExpression &) override {}
  virtual void Visit(const ExecuteMachineGroupExpression &) override {}
  virtual void Visit(const YieldExpression &) override {}
  virtual void Visit(const RandFloatExpression &) override {}
  virtual void Visit(const RandIntExpression &) override {}
  virtual void Visit(const ThisExpression &) override {}
  virtual void Visit(const UnresolvedAccessExpression &) override {}
  virtual void Visit(const TypeInitializerExpression &) override{};
  virtual void Visit(const AggregateContextExpression &) override{};
  virtual void Visit(const DebugExpression &) override {}

  virtual void VisitReference(const ArrayDereferenceExpression &) override {}
  virtual void VisitReference(const VariableRefExpression &) override {}
  virtual void VisitReference(const MemberAccessExpression &) override {}
  virtual void VisitReference(const ThisExpression &) override {}
  virtual void VisitReference(const UnresolvedAccessExpression &) override {}
};
}  // namespace FreeForm2

#endif
