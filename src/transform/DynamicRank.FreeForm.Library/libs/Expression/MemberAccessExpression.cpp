/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include "MemberAccessExpression.h"

#include <sstream>

#include "FreeForm2Assert.h"
#include "Mutation.h"
#include "SimpleExpressionOwner.h"
#include "StateMachine.h"
#include "StateMachineType.h"
#include "TypeUtil.h"
#include "Visitor.h"

namespace {
const FreeForm2::Expression &GetInitializer(
    const FreeForm2::StateMachineExpression &p_machine,
    const FreeForm2::CompoundType::Member &p_memberInfo) {
  const FreeForm2::TypeInitializerExpression &expr = p_machine.GetInitializer();
  for (const FreeForm2::TypeInitializerExpression::Initializer *iter =
           expr.BeginInitializers();
       iter != expr.EndInitializers(); ++iter) {
    if (&p_memberInfo == iter->m_member) {
      return *iter->m_initializer;
    }
  }
  FreeForm2::Unreachable(__FILE__, __LINE__);
}
}  // namespace

FreeForm2::MemberAccessExpression::MemberAccessExpression(
    const Annotations &p_annotations, const Expression &p_struct,
    const CompoundType::Member &p_memberInfo, size_t p_version)
    : Expression(Annotations(p_annotations.m_sourceLocation,
                             ValueBounds(*p_memberInfo.m_type))),
      m_struct(p_struct),
      m_memberInfo(p_memberInfo),
      m_version(p_version) {
  FF2_ASSERT(CompoundType::IsCompoundType(m_struct.GetType()));
  const CompoundType &compoundType =
      static_cast<const CompoundType &>(m_struct.GetType());
  FF2_ASSERT(compoundType.FindMember(p_memberInfo.m_name) != NULL);
}

const FreeForm2::TypeImpl &FreeForm2::MemberAccessExpression::GetType() const {
  return *m_memberInfo.m_type;
}

size_t FreeForm2::MemberAccessExpression::GetNumChildren() const { return 1; }

void FreeForm2::MemberAccessExpression::Accept(Visitor &p_visitor) const {
  size_t stackSize = p_visitor.StackSize();

  if (!p_visitor.AlternativeVisit(*this)) {
    m_struct.Accept(p_visitor);

    p_visitor.Visit(*this);
  }

  FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}

void FreeForm2::MemberAccessExpression::AcceptReference(
    Visitor &p_visitor) const {
  size_t stackSize = p_visitor.StackSize();

  m_struct.AcceptReference(p_visitor);

  p_visitor.VisitReference(*this);

  FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}

bool FreeForm2::MemberAccessExpression::IsConstant() const {
  if (m_struct.GetType().Primitive() == Type::StateMachine) {
    const StateMachineType &machine =
        static_cast<const StateMachineType &>(m_struct.GetType());
    boost::shared_ptr<const StateMachineExpression> ptr =
        machine.GetDefinition();
    return ptr != nullptr && m_memberInfo.m_type->IsConst() &&
           GetInitializer(*ptr, m_memberInfo).IsConstant();
  } else {
    return false;
  }
}

FreeForm2::ConstantValue FreeForm2::MemberAccessExpression::GetConstantValue()
    const {
  FF2_ASSERT(m_struct.GetType().Primitive() == Type::StateMachine);
  const StateMachineType &machine =
      static_cast<const StateMachineType &>(m_struct.GetType());
  boost::shared_ptr<const StateMachineExpression> ptr = machine.GetDefinition();
  return GetInitializer(*ptr, m_memberInfo).GetConstantValue();
}

const FreeForm2::Expression &FreeForm2::MemberAccessExpression::GetStruct()
    const {
  return m_struct;
}

const FreeForm2::CompoundType::Member &
FreeForm2::MemberAccessExpression::GetMemberInfo() const {
  return m_memberInfo;
}

size_t FreeForm2::MemberAccessExpression::GetVersion() const {
  return m_version;
}

FreeForm2::UnresolvedAccessExpression::UnresolvedAccessExpression(
    const Annotations &p_annotations, const Expression &p_object,
    const std::string &p_memberName, const TypeImpl &p_expectedType)
    : Expression(p_annotations),
      m_object(p_object),
      m_memberName(p_memberName),
      m_expectedType(p_expectedType) {}

const FreeForm2::TypeImpl &FreeForm2::UnresolvedAccessExpression::GetType()
    const {
  return m_expectedType;
}

size_t FreeForm2::UnresolvedAccessExpression::GetNumChildren() const {
  return 1;
}

void FreeForm2::UnresolvedAccessExpression::Accept(Visitor &p_visitor) const {
  size_t stackSize = p_visitor.StackSize();

  if (!p_visitor.AlternativeVisit(*this)) {
    m_object.Accept(p_visitor);

    p_visitor.Visit(*this);
  }

  FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}

void FreeForm2::UnresolvedAccessExpression::AcceptReference(
    Visitor &p_visitor) const {
  size_t stackSize = p_visitor.StackSize();

  m_object.AcceptReference(p_visitor);

  p_visitor.VisitReference(*this);

  FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}

const FreeForm2::Expression &FreeForm2::UnresolvedAccessExpression::GetObject()
    const {
  return m_object;
}

const std::string &FreeForm2::UnresolvedAccessExpression::GetMemberName()
    const {
  return m_memberName;
}
