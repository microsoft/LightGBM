/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include "StateMachineType.h"

#include <boost/make_shared.hpp>
#include "FreeForm2Assert.h"

FreeForm2::StateMachineType::StateMachineType(TypeManager &p_typeManager,
                                              const std::string &p_name,
                                              const Member *p_members,
                                              size_t p_numMembers,
                                              boost::weak_ptr<const StateMachineExpression> p_expr)
    : CompoundType(Type::StateMachine, false, &p_typeManager),
      m_name(p_name),
      m_expr(p_expr),
      m_numMembers(p_numMembers)
{
    if (m_numMembers > 0)
    {
        m_members[0] = p_members[0];

        // All subsequent members must be constructed.
        for (size_t i = 1; i < m_numMembers; i++)
        {
            m_members[i] = Member(p_members[i]);
        }
    }
}

FreeForm2::StateMachineType::~StateMachineType()
{
    // Only members > 1 need to be destructed; member 0 will be destructed
    // automatically.
    for (size_t i = 1; i < m_numMembers; i++)
    {
        m_members[i].Member::~Member();
    }
}

const std::string &
FreeForm2::StateMachineType::GetName() const
{
    return m_name;
}

const FreeForm2::CompoundType::Member *
FreeForm2::StateMachineType::FindMember(const std::string &p_name) const
{
    const MemberIterator end = EndMembers();
    for (MemberIterator iter = BeginMembers(); iter != end; ++iter)
    {
        if (iter->m_name == p_name)
        {
            return iter;
        }
    }
    return NULL;
}

FreeForm2::StateMachineType::MemberIterator
FreeForm2::StateMachineType::BeginMembers() const
{
    return m_members;
}

FreeForm2::StateMachineType::MemberIterator
FreeForm2::StateMachineType::EndMembers() const
{
    return BeginMembers() + m_numMembers;
}

size_t
FreeForm2::StateMachineType::GetMemberCount() const
{
    return m_numMembers;
}

const FreeForm2::TypeImpl &
FreeForm2::StateMachineType::AsConstType() const
{
    return *this;
}

const FreeForm2::TypeImpl &
FreeForm2::StateMachineType::AsMutableType() const
{
    return *this;
}

bool FreeForm2::StateMachineType::HasDefinition() const
{
    return !m_expr.expired();
}

boost::shared_ptr<const FreeForm2::StateMachineExpression>
FreeForm2::StateMachineType::GetDefinition() const
{
    return m_expr.lock();
}

bool FreeForm2::StateMachineType::IsSameSubType(const TypeImpl &p_type, bool p_ignoreConst) const
{
    FF2_ASSERT(p_type.Primitive() == Type::StateMachine);
    const StateMachineType &other = static_cast<const StateMachineType &>(p_type);
    return StateMachineType::GetName() == other.StateMachineType::GetName();
}
