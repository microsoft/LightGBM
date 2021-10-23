#pragma once

#ifndef FREEFORM2_OBJECT_TYPE_H
#define FREEFORM2_OBJECT_TYPE_H

#include "CompoundType.h"
#include <map>
#include <string>
#include <vector>

namespace FreeForm2
{
    // Object types are used to store information about external objects.
    class ObjectType : public CompoundType
    {
    public:
        // A struct that holds the information about members of an object.
        struct ObjectMember : public CompoundType::Member
        {
            // Constructor to initialize all members of the class.
            ObjectMember(const std::string& p_name,
                         const TypeImpl& p_type,
                         const std::string& p_externName);

            // Constructor used when frontend name matches the external name.
            ObjectMember(const std::string& p_name,
                         const TypeImpl& p_type);

            // The C++ name of the member.
            std::string m_externName;
        };

        // Get the name of this object type.
        virtual const std::string& GetName() const override;

        // Get the name of this object type.
        const std::string& GetExternName() const;

        // Find a member by name. Return value is NULL if the object
        // does not contain the specified member.
        virtual const ObjectType::ObjectMember* FindMember(const std::string& p_name) const override;

        // Create derived types based on this type.
        virtual const TypeImpl& AsConstType() const override;
        virtual const TypeImpl& AsMutableType() const override;
    private:
        // Create an ObjectType of the given function information.
        ObjectType(const std::string& p_name,
                   const std::string& p_externName,
                   const std::vector<ObjectMember>& p_members,
                   bool p_isConst,
                   TypeManager& p_typeManager);

        friend class TypeManager;

        // Compare subclass type data.
        virtual bool IsSameSubType(const TypeImpl& p_type, bool p_ignoreConst) const override;

        // Functions associated with this type.
        std::map<std::string, ObjectType::ObjectMember> m_members;

        // Name of this object in the frontend.
        std::string m_name;

        // Name of this object in the backend.
        std::string m_externName;
    };
}

#endif