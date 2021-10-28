/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#pragma once

#ifndef FREEFORM2_VISITOR_H
#define FREEFORM2_VISITOR_H

#include <basic_types.h>

#include <boost/noncopyable.hpp>

#include "FreeForm2Type.h"

namespace FreeForm2 {
class Expression;
class Allocation;
class AggregateContextExpression;
class ArrayDereferenceExpression;
class ArrayLengthExpression;
class ArrayLiteralExpression;
class BinaryOperatorExpression;
class BlockExpression;
class ComplexRangeLoopExpression;
class ConditionalExpression;
class ConvertToBoolExpression;
class ConvertToFloatExpression;
class ConvertToImperativeExpression;
class ConvertToIntExpression;
class ConvertToUInt64Expression;
class ConvertToInt32Expression;
class ConvertToUInt32Expression;
class DebugExpression;
class DeclarationExpression;
class DirectPublishExpression;
class ExecuteStreamRewritingStateMachineGroupExpression;
class ExecuteMachineExpression;
class ExecuteMachineGroupExpression;
class ExternExpression;
class FeatureRefExpression;
class FeatureSpecExpression;
class FeatureGroupSpecExpression;
class ForEachLoopExpression;
class FunctionExpression;
class FunctionCallExpression;
class ImportFeatureExpression;
class LetExpression;
class LiteralBoolExpression;
class LiteralFloatExpression;
class LiteralIntExpression;
class LiteralUInt64Expression;
class LiteralInt32Expression;
class LiteralUInt32Expression;
class LiteralStreamExpression;
class LiteralVoidExpression;
class LiteralWordExpression;
class LiteralInstanceHeaderExpression;
class MatchExpression;
class MatchOperatorExpression;
class MatchGuardExpression;
class MatchBindExpression;
class MemberAccessExpression;
class MutationExpression;
class PhiNodeExpression;
class PublishExpression;
class RandFloatExpression;
class RandIntExpression;
class RangeReduceExpression;
class ReturnExpression;
class SelectNthExpression;
class SelectRangeExpression;
class StateExpression;
class StateMachineExpression;
class StreamDataExpression;
class ThisExpression;
class TypeInitializerExpression;
class UnaryOperatorExpression;
class UnresolvedAccessExpression;
class UpdateStreamDataExpression;
class VariableRefExpression;
class YieldExpression;

// Visitor, an interface to implement the visitor pattern to generate code.
class Visitor : boost::noncopyable {
 public:
  // The visit method of the visitor pattern for each expression type.
  // The AlternativeVisit function for each type should return false if
  // a normal visitation is intended, and manually do the visitation return true
  // if it will manage the visitation itself.

  // The order of the Accept calls is:
  // * Children (0..number of children)
  // * Index
  virtual void Visit(const SelectNthExpression &) = 0;
  virtual bool AlternativeVisit(const SelectNthExpression &) { return false; }

  // The order of the Accept calls is:
  // * Start index
  // * Element count
  // * Source array
  virtual void Visit(const SelectRangeExpression &) = 0;
  virtual bool AlternativeVisit(const SelectRangeExpression &) { return false; }

  // The order of the Accept calls is:
  // * else
  // * then
  // * condition
  virtual void Visit(const ConditionalExpression &) = 0;
  virtual bool AlternativeVisit(const ConditionalExpression &) { return false; }

  // The order of the Accept calls is:
  // * Children (number of children-1..0)
  virtual void Visit(const ArrayLiteralExpression &) = 0;
  virtual bool AlternativeVisit(const ArrayLiteralExpression &) {
    return false;
  }

  // The order of the Accept calls is:
  // * Expressions (0..number of bound variables)
  // * The let value
  virtual void Visit(const LetExpression &) = 0;
  virtual bool AlternativeVisit(const LetExpression &) { return false; }

  // The order of the Accept calls is:
  // * Expressions (0..number of child expressions)
  virtual void Visit(const BlockExpression &) = 0;
  virtual bool AlternativeVisit(const BlockExpression &) { return false; }

  // The order of the Accept calls is:
  // * Children (number of children-1..0)
  virtual void Visit(const BinaryOperatorExpression &) = 0;
  virtual bool AlternativeVisit(const BinaryOperatorExpression &) {
    return false;
  }

  // The order of the Accept calls is:
  // * Initial value
  // * High value
  // * Low value
  // * Reduce value
  virtual void Visit(const RangeReduceExpression &) = 0;
  virtual bool AlternativeVisit(const RangeReduceExpression &) { return false; }

  virtual void Visit(const ForEachLoopExpression &) = 0;
  virtual bool AlternativeVisit(const ForEachLoopExpression &) { return false; }

  virtual void Visit(const ComplexRangeLoopExpression &) = 0;
  virtual bool AlternativeVisit(const ComplexRangeLoopExpression &) {
    return false;
  }

  // The order of the Accept calls is:
  // * l-value
  // * r-value
  virtual void Visit(const MutationExpression &) = 0;
  virtual bool AlternativeVisit(const MutationExpression &) { return false; }

  // The order of the Accept calls is:
  // * value to be matched
  // * pattern
  // * action
  virtual void Visit(const MatchExpression &) = 0;
  virtual bool AlternativeVisit(const MatchExpression &) { return false; }

  // The order of the Accept calls is:
  // * operands
  virtual void Visit(const MatchOperatorExpression &) = 0;
  virtual bool AlternativeVisit(const MatchOperatorExpression &) {
    return false;
  }

  virtual void Visit(const MatchGuardExpression &) = 0;
  virtual bool AlternativeVisit(const MatchGuardExpression &) { return false; }

  virtual void Visit(const MatchBindExpression &) = 0;
  virtual bool AlternativeVisit(const MatchBindExpression &) { return false; }

  // The order of the Accept calls is:
  // * constraints
  // virtual void Visit(const MatchWordExpression&) = 0;
  // virtual bool AlternativeVisit(const MatchWordExpression&)
  // {
  //     return false;
  // }

  // The order of the Accept calls is:
  // * struct
  virtual void Visit(const MemberAccessExpression &) = 0;
  virtual bool AlternativeVisit(const MemberAccessExpression &) {
    return false;
  }

  // The order of the Accept calls is:
  // * parameters (0..number of parameters)
  // * import feature parameters (0..number of import feature parameters)
  // * function body
  virtual void Visit(const FunctionExpression &) = 0;
  virtual bool AlternativeVisit(const FunctionExpression &) { return false; }

  // The order of the Accept calls is:
  // * function expression
  // * parameters (0..number of parameters)
  virtual void Visit(const FunctionCallExpression &) = 0;
  virtual bool AlternativeVisit(const FunctionCallExpression &) {
    return false;
  }

  // For all the unary expressions, the child accepts the visitor
  // before the call to Visit.
  virtual void Visit(const LiteralIntExpression &) = 0;
  virtual bool AlternativeVisit(const LiteralIntExpression &) { return false; }

  virtual void Visit(const LiteralUInt64Expression &) = 0;
  virtual bool AlternativeVisit(const LiteralUInt64Expression &) {
    return false;
  }

  virtual void Visit(const LiteralInt32Expression &) = 0;
  virtual bool AlternativeVisit(const LiteralInt32Expression &) {
    return false;
  }

  virtual void Visit(const LiteralUInt32Expression &) = 0;
  virtual bool AlternativeVisit(const LiteralUInt32Expression &) {
    return false;
  }

  virtual void Visit(const ArrayLengthExpression &) = 0;
  virtual bool AlternativeVisit(const ArrayLengthExpression &) { return false; }

  virtual void Visit(const ArrayDereferenceExpression &) = 0;
  virtual bool AlternativeVisit(const ArrayDereferenceExpression &) {
    return false;
  }

  virtual void Visit(const ConvertToFloatExpression &) = 0;
  virtual bool AlternativeVisit(const ConvertToFloatExpression &) {
    return false;
  }

  virtual void Visit(const ConvertToIntExpression &) = 0;
  virtual bool AlternativeVisit(const ConvertToIntExpression &) {
    return false;
  }

  virtual void Visit(const ConvertToUInt64Expression &) = 0;
  virtual bool AlternativeVisit(const ConvertToUInt64Expression &) {
    return false;
  }

  virtual void Visit(const ConvertToInt32Expression &) = 0;
  virtual bool AlternativeVisit(const ConvertToInt32Expression &) {
    return false;
  }

  virtual void Visit(const ConvertToUInt32Expression &) = 0;
  virtual bool AlternativeVisit(const ConvertToUInt32Expression &) {
    return false;
  }

  virtual void Visit(const ConvertToBoolExpression &) = 0;
  virtual bool AlternativeVisit(const ConvertToBoolExpression &) {
    return false;
  }

  virtual void Visit(const ConvertToImperativeExpression &) = 0;
  virtual bool AlternativeVisit(const ConvertToImperativeExpression &) {
    return false;
  }

  virtual void Visit(const DeclarationExpression &) = 0;
  virtual bool AlternativeVisit(const DeclarationExpression &) { return false; }

  virtual void Visit(const DirectPublishExpression &) = 0;
  virtual bool AlternativeVisit(const DirectPublishExpression &) {
    return false;
  }

  virtual void Visit(const ExternExpression &) = 0;
  virtual bool AlternativeVisit(const ExternExpression &) { return false; }

  virtual void Visit(const LiteralFloatExpression &) = 0;
  virtual bool AlternativeVisit(const LiteralFloatExpression &) {
    return false;
  }

  virtual void Visit(const LiteralBoolExpression &) = 0;
  virtual bool AlternativeVisit(const LiteralBoolExpression &) { return false; }

  virtual void Visit(const LiteralVoidExpression &) = 0;
  virtual bool AlternativeVisit(const LiteralVoidExpression &) { return false; }

  virtual void Visit(const LiteralStreamExpression &) = 0;
  virtual bool AlternativeVisit(const LiteralStreamExpression &) {
    return false;
  }

  virtual void Visit(const LiteralWordExpression &) = 0;
  virtual bool AlternativeVisit(const LiteralWordExpression &) { return false; }

  virtual void Visit(const LiteralInstanceHeaderExpression &) = 0;
  virtual bool AlternativeVisit(const LiteralInstanceHeaderExpression &) {
    return false;
  }

  virtual void Visit(const FeatureRefExpression &) = 0;
  virtual bool AlternativeVisit(const FeatureRefExpression &) { return false; }

  virtual void Visit(const UnaryOperatorExpression &) = 0;
  virtual bool AlternativeVisit(const UnaryOperatorExpression &) {
    return false;
  }

  virtual void Visit(const PhiNodeExpression &) = 0;
  virtual bool AlternativeVisit(const PhiNodeExpression &) { return false; }

  virtual void Visit(const PublishExpression &) = 0;
  virtual bool AlternativeVisit(const PublishExpression &) { return false; }

  virtual void Visit(const ReturnExpression &) = 0;
  virtual bool AlternativeVisit(const ReturnExpression &) { return false; }

  virtual void Visit(const FeatureSpecExpression &) = 0;
  virtual bool AlternativeVisit(const FeatureSpecExpression &) { return false; }

  virtual void Visit(const FeatureGroupSpecExpression &) = 0;
  virtual bool AlternativeVisit(const FeatureGroupSpecExpression &) {
    return false;
  }

  virtual void Visit(const StreamDataExpression &) = 0;
  virtual bool AlternativeVisit(const StreamDataExpression &) { return false; }

  virtual void Visit(const UpdateStreamDataExpression &) = 0;
  virtual bool AlternativeVisit(const UpdateStreamDataExpression &) {
    return false;
  }

  virtual void Visit(const VariableRefExpression &) = 0;
  virtual bool AlternativeVisit(const VariableRefExpression &) { return false; }

  virtual void Visit(const ImportFeatureExpression &) = 0;
  virtual bool AlternativeVisit(const ImportFeatureExpression &) {
    return false;
  }

  virtual void Visit(const StateExpression &) = 0;
  virtual bool AlternativeVisit(const StateExpression &) { return false; }

  virtual void Visit(const StateMachineExpression &) = 0;
  virtual bool AlternativeVisit(const StateMachineExpression &) {
    return false;
  }

  virtual void Visit(
      const ExecuteStreamRewritingStateMachineGroupExpression &) = 0;
  virtual bool AlternativeVisit(
      const ExecuteStreamRewritingStateMachineGroupExpression &) {
    return false;
  }

  virtual void Visit(const ExecuteMachineExpression &) = 0;
  virtual bool AlternativeVisit(const ExecuteMachineExpression &) {
    return false;
  }

  virtual void Visit(const ExecuteMachineGroupExpression &) = 0;
  virtual bool AlternativeVisit(const ExecuteMachineGroupExpression &) {
    return false;
  }

  virtual void Visit(const YieldExpression &) = 0;
  virtual bool AlternativeVisit(const YieldExpression &) { return false; }

  virtual void Visit(const RandFloatExpression &) = 0;
  virtual bool AlternativeVisit(const RandFloatExpression &) { return false; }

  virtual void Visit(const RandIntExpression &) = 0;
  virtual bool AlternativeVisit(const RandIntExpression &) { return false; }

  virtual void Visit(const ThisExpression &) = 0;
  virtual bool AlternativeVisit(const ThisExpression &) { return false; }

  virtual void Visit(const UnresolvedAccessExpression &) = 0;
  virtual bool AlternativeVisit(const UnresolvedAccessExpression &) {
    return false;
  }

  virtual void Visit(const TypeInitializerExpression &) = 0;
  virtual bool AlternativeVisit(const TypeInitializerExpression &) {
    return false;
  }

  virtual void Visit(const AggregateContextExpression &) = 0;
  virtual bool AlternativeVisit(const AggregateContextExpression &) {
    return false;
  }

  virtual void Visit(const DebugExpression &) = 0;
  virtual bool AlternativeVisit(const DebugExpression &) { return false; }

  // VisitReference functions act like regular visit functions, except
  // that the quantity generated should reference the expressed value,
  // instead of the value itself.  This arrangement, of strongly
  // separating reference visitation from value vistation, forces us to be
  // careful about reference/value distinctions, which seems to be wise.
  //
  // Note that we drawing a distinction between reference and value return
  // at the class level would involve more duplication (duplicating
  // classes, as well as Visit methods).
  //
  // Another alternative (with less duplication, but seems
  // dangerously unsafe) is to set flags during iteration to indicate
  // whether reference/value return is expected.
  virtual void VisitReference(const ArrayDereferenceExpression &) = 0;
  virtual bool AlternativeVisitReference(const ArrayDereferenceExpression &) {
    return false;
  }

  virtual void VisitReference(const VariableRefExpression &) = 0;
  virtual void VisitReference(const MemberAccessExpression &) = 0;
  virtual void VisitReference(const ThisExpression &) = 0;
  virtual void VisitReference(const UnresolvedAccessExpression &) = 0;

  // These two methods allow us to check on the correctness of a common
  // idiom, which is to use a stack to keep results from subexpressions.
  // Bugs affecting the size of this stack are irritating to track, since
  // they aren't caught until significantly later than they occur.
  // Override these to benefit from stack size checking during visitation.
  virtual size_t StackSize() const { return 0; }
  virtual size_t StackIncrement() const { return 0; }
};
}  // namespace FreeForm2

#endif
