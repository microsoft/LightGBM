/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include "UniformExpressionVisitor.h"

#include "Allocation.h"
#include "ArrayDereferenceExpression.h"
#include "ArrayLength.h"
#include "ArrayLiteralExpression.h"
#include "BlockExpression.h"
#include "Conditional.h"
#include "ConvertExpression.h"
#include "DebugExpression.h"
#include "Declaration.h"
#include "Expression.h"
#include "Extern.h"
#include "FeatureSpec.h"
#include "Function.h"
#include "LetExpression.h"
#include "LiteralExpression.h"
#include "Match.h"
#include "MemberAccessExpression.h"
#include "Mutation.h"
#include "OperatorExpression.h"
#include "PhiNode.h"
#include "Publish.h"
#include "RandExpression.h"
#include "RangeReduceExpression.h"
#include "RefExpression.h"
#include "SelectNth.h"
#include "StateMachine.h"
#include "StreamData.h"

void FreeForm2::UniformExpressionVisitor::Visit(
    const SelectNthExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const SelectRangeExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const ConditionalExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const ArrayLiteralExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(const LetExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(const BlockExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const BinaryOperatorExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const RangeReduceExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const ForEachLoopExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const ComplexRangeLoopExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const MutationExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(const MatchExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const MatchOperatorExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const MatchGuardExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const MatchBindExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const MemberAccessExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const ArrayLengthExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const ArrayDereferenceExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const ConvertToFloatExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const ConvertToIntExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const ConvertToUInt64Expression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const ConvertToInt32Expression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const ConvertToUInt32Expression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const ConvertToBoolExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const ConvertToImperativeExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const DeclarationExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const DirectPublishExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const ExternExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const FunctionExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const FunctionCallExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const LiteralIntExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const LiteralUInt64Expression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const LiteralInt32Expression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const LiteralUInt32Expression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const LiteralFloatExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const LiteralBoolExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const LiteralVoidExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const LiteralStreamExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const LiteralWordExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const LiteralInstanceHeaderExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const FeatureRefExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const UnaryOperatorExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const FeatureSpecExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const FeatureGroupSpecExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const PhiNodeExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const PublishExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const ReturnExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const StreamDataExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const UpdateStreamDataExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const VariableRefExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const ImportFeatureExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(const StateExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const StateMachineExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const ExecuteStreamRewritingStateMachineGroupExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const ExecuteMachineExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const ExecuteMachineGroupExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(const YieldExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const RandFloatExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const RandIntExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(const ThisExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const UnresolvedAccessExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const TypeInitializerExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(
    const AggregateContextExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::VisitReference(
    const ArrayDereferenceExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::VisitReference(
    const VariableRefExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::VisitReference(
    const MemberAccessExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::VisitReference(
    const ThisExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::VisitReference(
    const UnresolvedAccessExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}

void FreeForm2::UniformExpressionVisitor::Visit(const DebugExpression &p_expr) {
  const Expression &expr = p_expr;
  Visit(expr);
}
