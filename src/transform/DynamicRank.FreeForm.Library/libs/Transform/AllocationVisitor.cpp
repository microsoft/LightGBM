/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include "AllocationVisitor.h"

#include "Allocation.h"
#include "ArrayLiteralExpression.h"
#include "Declaration.h"
#include "Extern.h"
#include "FeatureSpec.h"
#include "FreeForm2Assert.h"
#include "LiteralExpression.h"
#include "StateMachine.h"

FreeForm2::AllocationVisitor::AllocationVisitor(const Expression &p_exp) {
  p_exp.Accept(*this);
}

const FreeForm2::AllocationVisitor::AllocationVector &
FreeForm2::AllocationVisitor::GetAllocations() const {
  return m_allocations;
}

void FreeForm2::AllocationVisitor::Visit(const ArrayLiteralExpression &p_expr) {
  if (m_allocationIds.find(p_expr.GetId()) == m_allocationIds.end()) {
    m_allocations.push_back(boost::shared_ptr<Allocation>(new Allocation(
        Allocation::ArrayLiteral, p_expr.GetId(), p_expr.GetType())));
    m_allocationIds.insert(p_expr.GetId());
  }
}

void FreeForm2::AllocationVisitor::Visit(
    const LiteralStreamExpression &p_expr) {
  if (m_allocationIds.find(p_expr.GetId()) == m_allocationIds.end()) {
    m_allocations.push_back(boost::shared_ptr<Allocation>(
        new Allocation(Allocation::LiteralStream, p_expr.GetId(),
                       p_expr.GetType(), p_expr.GetNumChildren())));
    m_allocationIds.insert(p_expr.GetId());
  }
}

void FreeForm2::AllocationVisitor::Visit(const LiteralWordExpression &p_expr) {
  if (m_allocationIds.find(p_expr.GetId()) == m_allocationIds.end()) {
    m_allocations.push_back(boost::shared_ptr<Allocation>(new Allocation(
        Allocation::LiteralWord, p_expr.GetId(), p_expr.GetType())));
    m_allocationIds.insert(p_expr.GetId());
  }
}

void FreeForm2::AllocationVisitor::Visit(const DeclarationExpression &p_expr) {
  if (m_allocationIds.find(p_expr.GetId()) == m_allocationIds.end()) {
    m_allocations.push_back(boost::shared_ptr<Allocation>(new Allocation(
        Allocation::Declaration, p_expr.GetId(), p_expr.GetDeclaredType())));
    m_allocationIds.insert(p_expr.GetId());
  }
}

void FreeForm2::AllocationVisitor::Visit(const ExternExpression &p_expr) {
  if (p_expr.GetType().Primitive() == Type::Array) {
    if (m_allocationIds.find(p_expr.GetId()) == m_allocationIds.end()) {
      m_allocations.push_back(boost::shared_ptr<Allocation>(new Allocation(
          Allocation::ExternArray, p_expr.GetId(), p_expr.GetType())));
      m_allocationIds.insert(p_expr.GetId());
    }
  }
}

void FreeForm2::AllocationVisitor::Visit(
    const ImportFeatureExpression &p_expr) {
  if (p_expr.GetType().Primitive() == Type::Array) {
    m_allocations.push_back(boost::shared_ptr<Allocation>(new Allocation(
        Allocation::FeatureArray, p_expr.GetId(), p_expr.GetType())));
  } else {
    m_allocations.push_back(boost::shared_ptr<Allocation>(new Allocation(
        Allocation::Declaration, p_expr.GetId(), p_expr.GetType())));
  }
  m_allocationIds.insert(p_expr.GetId());
}

void FreeForm2::AllocationVisitor::Visit(const StateExpression &p_expr) {
  // Don't actually add any allocations for state expressions, but visit
  // its actions and leaving actions for transitions, in case they allocate
  // something.
  for (auto &action : p_expr.m_actions) {
    action.m_action->Accept(*this);
  }

  for (auto &transition : p_expr.m_transitions) {
    if (transition.m_leavingAction) {
      transition.m_leavingAction->Accept(*this);
    }
  }
}

void FreeForm2::AllocationVisitor::Visit(
    const ExecuteMachineExpression &p_expr) {
  FF2_ASSERT(p_expr.GetMachine().GetType().Primitive() == Type::StateMachine);

  const StateMachineType &machineType =
      dynamic_cast<const StateMachineType &>(p_expr.GetMachine().GetType());

  machineType.GetDefinition()->Accept(*this);
}

bool FreeForm2::AllocationVisitor::AlternativeVisit(
    const FunctionExpression &p_expr) {
  // Allocations should not cross function boundaries.
  return true;
}

bool FreeForm2::AllocationVisitor::AlternativeVisit(
    const ExecuteStreamRewritingStateMachineGroupExpression &p_expr) {
  // Allocations should not cross stream rewriting machine group boundaries.
  return true;
}