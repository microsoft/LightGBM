/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include "Mutation.h"

#include <boost/foreach.hpp>
#include <set>
#include <sstream>

#include "FreeForm2Assert.h"
#include "StateMachineType.h"
#include "StructType.h"
#include "TypeUtil.h"
#include "Visitor.h"

namespace {
void DeleteAlloc(FreeForm2::TypeInitializerExpression *p_delete) {
  // Explicitly invoke the destructor.
  p_delete->~TypeInitializerExpression();

  // Cast the char*, as the memory was allocated as char[].
  char *mem = reinterpret_cast<char *>(p_delete);
  delete[] mem;
}
}  // namespace

FreeForm2::MutationExpression::MutationExpression(
    const Annotations &p_annotations, const Expression &p_lvalue,
    const Expression &p_rvalue)
    : Expression(p_annotations), m_lvalue(p_lvalue), m_rvalue(p_rvalue) {}

void FreeForm2::MutationExpression::Accept(Visitor &p_visitor) const {
  size_t stackSize = p_visitor.StackSize();

  if (!p_visitor.AlternativeVisit(*this)) {
    try {
      m_lvalue.AcceptReference(p_visitor);
    } catch (const std::exception &) {
      std::ostringstream err;
      err << "Invalid l-value in mutation expression";
      throw ParseError(err.str(), GetSourceLocation());
    }

    m_rvalue.Accept(p_visitor);

    p_visitor.Visit(*this);
  }

  FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}

const FreeForm2::TypeImpl &FreeForm2::MutationExpression::GetType() const {
  const TypeImpl &left = m_lvalue.GetType();
  const TypeImpl &right = m_rvalue.GetType();

  if (!TypeUtil::IsAssignable(left, right)) {
    std::ostringstream err;
    err << "Mismatched types in assignment (" << left << " and " << right
        << ")";
    throw ParseError(err.str(), GetSourceLocation());
  }

  if (left.Primitive() == Type::Array) {
    std::ostringstream err;
    err << "Can't assign types that are not of fixed size (such as arrays)";
    throw ParseError(err.str(), GetSourceLocation());
  }

  if (left.IsConst()) {
    std::ostringstream err;
    err << "Can't assign to constant types";
    throw ParseError(err.str(), GetSourceLocation());
  }

  return TypeImpl::GetVoidInstance();
}

size_t FreeForm2::MutationExpression::GetNumChildren() const { return 2; }

const FreeForm2::Expression &FreeForm2::MutationExpression::GetLeftValue()
    const {
  return m_lvalue;
}

const FreeForm2::Expression &FreeForm2::MutationExpression::GetRightValue()
    const {
  return m_rvalue;
}

FreeForm2::TypeInitializerExpression::TypeInitializerExpression(
    const Annotations &p_annotations, const CompoundType &p_type,
    const Initializer *p_initializers, size_t p_numInitializers)
    : Expression(p_annotations),
      m_type(p_type),
      m_numInitializers(p_numInitializers) {
  memcpy(m_initializers, p_initializers,
         sizeof(Initializer) * m_numInitializers);
  ValidateMembers();
}

boost::shared_ptr<FreeForm2::TypeInitializerExpression>
FreeForm2::TypeInitializerExpression::Alloc(const Annotations &p_annotations,
                                            const CompoundType &p_type,
                                            const Initializer *p_initializers,
                                            size_t p_numInitializers) {
  const size_t memSize =
      sizeof(TypeInitializerExpression) +
      sizeof(Initializer) * (std::max(p_numInitializers, (size_t)1ULL) - 1);
  char *mem = NULL;
  try {
    mem = new char[memSize];
    return boost::shared_ptr<TypeInitializerExpression>(
        new (mem) TypeInitializerExpression(p_annotations, p_type,
                                            p_initializers, p_numInitializers));
  } catch (...) {
    delete[] mem;
    throw;
  }
}

void FreeForm2::TypeInitializerExpression::ValidateMembers() const {
  // Collect all member names in the type into a set.
  std::set<std::string> names;
  if (m_type.Primitive() == Type::Struct) {
    const StructType &type = static_cast<const StructType &>(m_type);
    BOOST_FOREACH (const StructType::MemberInfo &member, type.GetMembers()) {
      names.insert(member.m_name);
    }
  } else {
    FF2_ASSERT(m_type.Primitive() == Type::StateMachine);
    const StateMachineType &type =
        static_cast<const StateMachineType &>(m_type);
    for (const StructType::Member *iter = type.BeginMembers();
         iter != type.EndMembers(); ++iter) {
      names.insert(iter->m_name);
    }
  }

  // Search for names not being initialized.
  for (const Initializer *iter = BeginInitializers(); iter != EndInitializers();
       ++iter) {
    const TypeImpl &memberType = *iter->m_member->m_type;
    const TypeImpl &initType = iter->m_initializer->GetType();
    if (!TypeUtil::IsAssignable(memberType, initType)) {
      std::ostringstream err;
      err << "Mismatched types in initializer (" << memberType << " and "
          << initType << ")";
      throw ParseError(err.str(), GetSourceLocation());
    }

    FF2_ASSERT(iter != NULL && iter->m_member != NULL);
    std::set<std::string>::iterator find = names.find(iter->m_member->m_name);
    FF2_ASSERT(find != names.end());
    names.erase(find);
  }

  if (!names.empty()) {
    std::ostringstream err;
    err << "all members must be initialized; missing: ";
    BOOST_FOREACH (const std::string &name, names) { err << name << " "; }
    throw ParseError(err.str(), GetSourceLocation());
  }
}

void FreeForm2::TypeInitializerExpression::Accept(Visitor &p_visitor) const {
  const size_t stackSize = p_visitor.StackSize();

  if (!p_visitor.AlternativeVisit(*this)) {
    for (const Initializer *iter = BeginInitializers();
         iter != EndInitializers(); ++iter) {
      iter->m_initializer->Accept(p_visitor);
    }

    p_visitor.Visit(*this);
  }

  FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}

const FreeForm2::TypeImpl &FreeForm2::TypeInitializerExpression::GetType()
    const {
  return m_type;
}

size_t FreeForm2::TypeInitializerExpression::GetNumChildren() const {
  return m_numInitializers;
}

const FreeForm2::TypeInitializerExpression::Initializer *
FreeForm2::TypeInitializerExpression::BeginInitializers() const {
  return m_initializers;
}

const FreeForm2::TypeInitializerExpression::Initializer *
FreeForm2::TypeInitializerExpression::EndInitializers() const {
  return m_initializers + m_numInitializers;
}
