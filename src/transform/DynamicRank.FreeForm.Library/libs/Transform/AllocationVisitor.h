/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#pragma once

#ifndef FREEFORM2_ARRAY_ALLOCATION_VISITOR_H
#define FREEFORM2_ARRAY_ALLOCATION_VISITOR_H

#include <boost/shared_ptr.hpp>
#include <set>
#include <vector>

#include "Expression.h"
#include "NoOpVisitor.h"

namespace FreeForm2 {
class AllocationVisitor : public NoOpVisitor {
 public:
  typedef std::vector<boost::shared_ptr<Allocation> > AllocationVector;

  // Create a new instance.
  AllocationVisitor(const Expression &p_exp);

  // Return the list of allocations for this expression tree.
  const AllocationVector &GetAllocations() const;

  // Visits the ArrayLiteralExpression and adds an Array Allocation
  // to the list.
  virtual void Visit(const ArrayLiteralExpression &p_expr) override;

  // Visits the LiteralStreamExpression and adds an
  // LiteralStreamAllocationExpression to the list.
  virtual void Visit(const LiteralStreamExpression &p_expr) override;

  // Visits the LiteralWordExpression and adds an LiteralWord Allocation
  // to the list.
  virtual void Visit(const LiteralWordExpression &p_expr) override;

  // Visits the DeclarationExpression and adds a
  // Declaration Allocation to the list.
  virtual void Visit(const DeclarationExpression &p_expr) override;

  // Visits the ExternExpression and adds a Declaration Allocation
  // and an ArrayLiteralAllocationExpression if applicable.
  virtual void Visit(const ExternExpression &p_expr) override;

  // Visits the ImportFeatureExpression and adds either an array literal
  // or declaration allocation depending on the type.
  virtual void Visit(const ImportFeatureExpression &p_expr) override;

  // Visits the ExecuteMachineExpression and StateExpression so the allocations
  // done within the actions or transitions of a state machine can also be
  // considered for allocations.
  virtual void Visit(const StateExpression &p_expr) override;
  virtual void Visit(const ExecuteMachineExpression &p_expr) override;

  // Skips the visitation of FunctionExpressions.
  virtual bool AlternativeVisit(const FunctionExpression &p_expr) override;

  // Skip the visitation of ExecuteStreamRewritingStateMachineGroupExpressions.
  virtual bool AlternativeVisit(
      const ExecuteStreamRewritingStateMachineGroupExpression &p_expr) override;

 private:
  // Holds all the array allocation expressions.
  AllocationVector m_allocations;

  // A set of allocation IDs created to avoid duplicated allocations.
  // Duplcates are possible in, for example, array-based range reduce
  // expressions, in which the array dereference and array length in the
  // loop reference the same ArrayLiteralExpression.
  std::set<VariableID> m_allocationIds;
};
}  // namespace FreeForm2

#endif
