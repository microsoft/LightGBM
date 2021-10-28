/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#pragma once

#ifndef FREEFORM2_RANGE_REDUCE_EXPRESSION_H
#define FREEFORM2_RANGE_REDUCE_EXPRESSION_H

#include "Expression.h"

// (range-reduce curr 0 10 prev 0.0 (+ prev curr))

namespace FreeForm2 {
class SimpleExpressionOwner;
class TypeManager;

// A range-reduce expression generates a loop that reduces integer values
// in a given range to a final quantity.
class RangeReduceExpression : public Expression {
 public:
  // Construct a range-reduce expression to loop over a range specified
  // as an Expression pair; the
  RangeReduceExpression(const Annotations &p_annotations,
                        const Expression &p_low, const Expression &p_high,
                        const Expression &p_initial, const Expression &p_reduce,
                        VariableID p_stepId, VariableID p_reduceId);

  // Methods inherited from Expression.
  virtual size_t GetNumChildren() const override;
  virtual void Accept(Visitor &p_visitor) const override;
  virtual const TypeImpl &GetType() const override;

  // Expose all children nodes for the various AlternativeVisit methods.
  const Expression &GetLow() const;
  const Expression &GetHigh() const;
  const Expression &GetInitial() const;
  const Expression &GetReduceExpression() const;
  VariableID GetReduceId() const;
  VariableID GetStepId() const;

 private:
  // Infer the resulting type of the range-reduce expression.
  const TypeImpl &InferType() const;

  // Low range expression.
  const Expression &m_low;

  // High range expression.
  const Expression &m_high;

  // ID of the step variable.
  VariableID m_stepId;

  // Initial reduction value.
  const Expression &m_initial;

  // Reduction expression.
  const Expression &m_reduce;

  // ID of the reduction variable
  VariableID m_reduceId;

  // The type of this expression.
  const TypeImpl &m_type;
};

// A for-each loop expression represents an iterative loop with a beginning
// value, an ending value, a next expression, and a loop body. As a
// required precondition, successive evaluations of the next expression
// will eventually result in the expression [current == end] being true.
class ForEachLoopExpression : public Expression {
 public:
  // Loop hints allow the backend to optimize its implementation of the
  // loop.
  enum LoopHint { NoHint, HintStepIncreasing, HintStepDecreasing };

  // Create a for-each loop over a set of bounds. At the end of each
  // evaluation of body, the iterator variable is assigned to the result
  // of the p_next expression. The loop breaks when the iterator variable
  // is equal to the second member of the bounds pair.
  ForEachLoopExpression(
      const Annotations &p_annotations,
      const std::pair<const Expression *, const Expression *> &p_bounds,
      const Expression &p_next, const Expression &p_body,
      VariableID p_iteratorId, size_t p_version, LoopHint p_hint,
      TypeManager &p_typeManager);

  // Methods inherited from Expression.
  virtual size_t GetNumChildren() const override;
  virtual void Accept(Visitor &p_visitor) const override;
  virtual const TypeImpl &GetType() const override;

  // Accessor methods
  const Expression &GetBegin() const;
  const Expression &GetEnd() const;
  const Expression &GetNext() const;
  const Expression &GetBody() const;
  const TypeImpl &GetIteratorType() const;
  VariableID GetIteratorId() const;
  size_t GetVersion() const;
  LoopHint GetHint() const;

 private:
  // The beginning and ending bounds of the loop.
  const Expression &m_begin;
  const Expression &m_end;

  // This expression is evaluated after each iteration to get the next
  // iterator value.
  const Expression &m_next;

  // The loop body.
  const Expression &m_body;

  // The type of the iterator variable. This type is compatible with the
  // bounds expressions and the next expression.
  const TypeImpl *m_iteratorType;

  // The variable ID of the iterator variable.
  VariableID m_iteratorId;

  // A unique version number associated with a particular
  // value for this variable.
  const size_t m_version;

  // Implementation loop hint.
  LoopHint m_hint;
};

// A complex range loop is a loop with the following properties:
//  - The loop has a range [low, high) and a step, which are all integers.
//  - The loop includes preconditions that assert the safety of the loop,
//     specifically testing under/overflow.
//  - The following is an expected precondition of the loop:
//     step > 0 == high > low && step != 0.
//  - If all preconditions are met, the loop will execute at least once.
//  - At each iteration of the loop, there exists a variable i such that
//     i = low + step * j, where j is the number of times the loop body has
//     executed.
//  - The loop condition follows post-test execution pattern.
// This is a more complex loop than the above structures, as it is not the
// case that the iterative variable ever be equal to the high value; the
// loop will break before passing the high value of the range.
class ComplexRangeLoopExpression : public Expression {
 public:
  // Create a complex range loop expression with correct loop conditions
  // according to the properties above. This method derives the
  // precondition and loop condition from the given data.
  static const ComplexRangeLoopExpression &Create(
      const Annotations &p_annotations,
      const std::pair<const Expression *, const Expression *> &p_range,
      const Expression &p_step, const Expression &p_body,
      const Expression &p_loopVar, VariableID p_loopVarId, size_t p_version,
      SimpleExpressionOwner &p_owner, TypeManager &p_typeManager);

  // Create a complex range loop specifying all properties of the loop.
  ComplexRangeLoopExpression(
      const Annotations &p_annotations,
      const std::pair<const Expression *, const Expression *> &p_range,
      const Expression &p_step, const Expression &p_body,
      const Expression &p_precondition, const Expression &p_loopCondition,
      const TypeImpl &p_stepType, VariableID p_stepId, size_t p_version);

  // Methods inherited from Expression.
  virtual size_t GetNumChildren() const override;
  virtual void Accept(Visitor &p_visitor) const override;
  virtual const TypeImpl &GetType() const override;

  // Accessor methods
  const Expression &GetPrecondition() const;
  const Expression &GetLow() const;
  const Expression &GetHigh() const;
  const Expression &GetStep() const;
  const Expression &GetBody() const;
  const Expression &GetLoopCondition() const;
  const TypeImpl &GetStepType() const;
  VariableID GetStepId() const;
  size_t GetVersion() const;

 private:
  // The low and high bounds of the range.
  const Expression &m_low;
  const Expression &m_high;

  // This expression is evaluated after each iteration to get the next
  // iterator value.
  const Expression &m_step;

  // The loop body.
  const Expression &m_body;

  // These expressions are generated by the loop expression. The
  // precondition is evaluated before the loop, and the loop condition
  // is evaluated at every
  const Expression &m_precondition;
  const Expression &m_loopCondition;

  // The type of the iterator variable.
  const TypeImpl &m_stepType;

  // The variable ID of the iterator variable.
  VariableID m_stepId;

  // A unique version number associated with a particular
  // value for this variable.
  const size_t m_version;
};
};  // namespace FreeForm2

#endif
