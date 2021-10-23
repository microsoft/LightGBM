#include "ObjectType.h"

#include <boost/foreach.hpp>
#include "FreeForm2Assert.h"
#include "TypeManager.h"

FreeForm2::ObjectType::ObjectMember::ObjectMember(const std::string& p_name,
                                                  const TypeImpl& p_type,
                                                  const std::string& p_externName)
    : CompoundType::Member(p_name, p_type),
      m_externName(p_externName)
{
}

FreeForm2::ObjectType::ObjectMember::ObjectMember(const std::string& p_name,
                                                  const TypeImpl& p_type)
    : CompoundType::Member(p_name, p_type),
      m_externName(p_name)
{
}


FreeForm2::ObjectType::ObjectType(const std::string& p_name, 
                                  const std::string& p_externName, 
                                  const std::vector<ObjectType::ObjectMember>& p_members, 
                                  bool p_isConst, 
                                  TypeManager& p_typeManager)
    : CompoundType(Type::Object, p_isConst, &p_typeManager),
      m_name(p_name),
      m_externName(p_externName)
{
    BOOST_FOREACH (ObjectType::ObjectMember member, p_members)
    {
        // Verify that all names are unique.
        FF2_ASSERT(m_members.find(member.m_name) == m_members.end());
        m_members.insert(std::make_pair(member.m_name, member));
    }
}


const std::string&
FreeForm2::ObjectType::GetName() const
{
    return m_name;
}


const std::string&
FreeForm2::ObjectType::GetExternName() const
{
    return m_externName;
}


const FreeForm2::ObjectType::ObjectMember*
FreeForm2::ObjectType::FindMember(const std::string& p_name) const
{
    std::map<std::string, ObjectType::ObjectMember>::const_iterator member = m_members.find(p_name);
    return member != m_members.end() ? &member->second : NULL;
}


const FreeForm2::TypeImpl& 
FreeForm2::ObjectType::AsConstType() const
{
    if (IsConst())
    {
        return *this;
    }
    else
    {
        FF2_ASSERT(GetTypeManager() != NULL);    
        
        std::vector<ObjectType::ObjectMember> members;
        for (std::map<std::string, ObjectType::ObjectMember>::const_iterator memberIterator = m_members.begin();
             memberIterator !=  m_members.end();
             ++memberIterator)
        {
            members.push_back(memberIterator->second);
        }

        return GetTypeManager()->GetObjectType(GetName(), GetExternName(), members, true);
    }
}


const FreeForm2::TypeImpl& 
FreeForm2::ObjectType::AsMutableType() const
{
    if (!IsConst())
    {
        return *this;
    }
    else
    {
        FF2_ASSERT(GetTypeManager() != NULL);

        std::vector<ObjectType::ObjectMember> members;
        for (std::map<std::string, ObjectType::ObjectMember>::const_iterator memberIterator = m_members.begin();
             memberIterator !=  m_members.end();
             ++memberIterator)
        {
            members.push_back(memberIterator->second);
        }

        return GetTypeManager()->GetObjectType(GetName(), GetExternName(), members, false);
    }
}


bool
FreeForm2::ObjectType::IsSameSubType(const TypeImpl& p_other, bool p_ignoreConst) const
{
    FF2_ASSERT(p_other.Primitive() == Type::Object);
    const ObjectType& other = static_cast<const ObjectType&>(p_other);

    return GetName() == other.GetName() 
           && GetExternName() == other.GetExternName();
}