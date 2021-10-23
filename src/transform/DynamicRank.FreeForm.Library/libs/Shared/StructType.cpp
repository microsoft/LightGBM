#include "StructType.h"

#include <boost/foreach.hpp>
#include "FreeForm2Assert.h"
#include <sstream>
#include "TypeManager.h"

FreeForm2::StructType::MemberInfo::MemberInfo(const std::string& p_name,
                                              const TypeImpl& p_type,
                                              const std::string& p_externName,
                                              size_t p_offset,
                                              size_t p_size)
    : Member(p_name, p_type),
      m_externName(p_externName),
      m_offset(p_offset),
      m_size(p_size)
{
}


bool
FreeForm2::StructType::MemberInfo::operator==(const MemberInfo& p_other) const
{
    return (m_name == p_other.m_name
            && m_externName == p_other.m_externName
            && m_offset == p_other.m_offset
            && m_size == p_other.m_size
            && (*m_type == *p_other.m_type));
}


bool
FreeForm2::StructType::MemberInfo::operator!=(const MemberInfo& p_other) const
{
    return !(*this == p_other);
}


FreeForm2::StructType::StructType(const std::string& p_name,
                                  const std::string& p_externName,
                                  const std::vector<MemberInfo>& p_members,
                                  bool p_isConst,
                                  TypeManager& p_typeManager)
    : CompoundType(Type::Struct, p_isConst, &p_typeManager),
      m_name(p_name),
      m_externName(p_externName)
{
    m_members.reserve(p_members.size());
    BOOST_FOREACH (StructType::MemberInfo member, p_members)
    {
        const TypeImpl& memberType 
            = p_isConst ? member.m_type->AsConstType() : member.m_type->AsMutableType();
        m_members.push_back(MemberInfo(member.m_name, 
                                       memberType, 
                                       member.m_externName, 
                                       member.m_offset, 
                                       member.m_size));

        // Verify that all names are unique.
        FF2_ASSERT(m_memberMapping.find(m_members.back().m_name) == m_memberMapping.end());
        m_memberMapping.insert(std::make_pair(m_members.back().m_name, &m_members.back()));
    }
}


const FreeForm2::CompoundType::Member*
FreeForm2::StructType::FindMember(const std::string& p_name) const
{
    return FindStructMember(p_name);
}


const FreeForm2::StructType::MemberInfo*
FreeForm2::StructType::FindStructMember(const std::string& p_name) const
{
    std::map<std::string, const StructType::MemberInfo*>::const_iterator member
        = m_memberMapping.find(p_name);
    return (member != m_memberMapping.end()) ? member->second : NULL;
}


const std::vector<FreeForm2::StructType::MemberInfo>&
FreeForm2::StructType::GetMembers() const
{
    return m_members;
}


const std::string&
FreeForm2::StructType::GetName() const
{
    return m_name;
}


const std::string&
FreeForm2::StructType::GetExternName() const
{
    return m_externName;
}


std::string
FreeForm2::StructType::GetString() const
{
    std::ostringstream out;
    out << "struct " << GetName() << "{";

    bool first = true;

    BOOST_FOREACH (StructType::MemberInfo member, GetMembers())
    {
        if (!first)
        {
            out << ", ";
        }

        out << *member.m_type << " " << member.m_name;

        first = false;
    }

    out << "}";

    return out.str();
}

const FreeForm2::TypeImpl& 
FreeForm2::StructType::AsConstType() const
{
    if (IsConst())
    {
        return *this;
    }
    else
    {
        FF2_ASSERT(GetTypeManager() != NULL);
        return GetTypeManager()->GetStructType(GetName(), GetExternName(), GetMembers(), true);
    }
}


const FreeForm2::TypeImpl& 
FreeForm2::StructType::AsMutableType() const
{
    if (!IsConst())
    {
        return *this;
    }
    else
    {
        FF2_ASSERT(GetTypeManager() != NULL);
        return GetTypeManager()->GetStructType(GetName(), GetExternName(), GetMembers(), false);
    }
}


bool
FreeForm2::StructType::IsSameSubType(const TypeImpl& p_other, bool p_ignoreConst) const
{
    FF2_ASSERT(p_other.Primitive() == Type::Struct);
    const StructType& other = static_cast<const StructType&>(p_other);

    if (GetMembers().size() != other.GetMembers().size())
    {
        return false;
    }

    for (size_t i = 0; i < GetMembers().size(); i++)
    {
        const MemberInfo& m1 = GetMembers()[i];
        const MemberInfo& m2 = other.GetMembers()[i];
        if (!m1.m_type->IsSameAs(*m2.m_type, p_ignoreConst)
            || m1.m_externName != m2.m_externName
            || m1.m_name != m2.m_name
            || m1.m_offset != m2.m_offset
            || m1.m_size != m2.m_size)
        {
            return false;
        }
    }

    return GetName() == other.GetName() 
           && GetExternName() == other.GetExternName();
}

