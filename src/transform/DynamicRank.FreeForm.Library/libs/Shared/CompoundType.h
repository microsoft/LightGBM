#pragma once

#ifndef FREEFORM2_COMPOUND_TYPE_H
#define FREEFORM2_COMPOUND_TYPE_H

#include "FreeForm2Type.h"
#include <string>
#include "TypeImpl.h"

namespace FreeForm2
{
    class TypeManager;

    // A compound type is any type which contains named instantiations of other
    // types. This class provides a mechanism for looking up member information
    // by name.
    class CompoundType : public TypeImpl
    {
    public:
        // A member represents a single named entity contained within a 
        // compound type.
        struct Member
        {
        public:
            // Construct a member with a name and type.
            Member(const std::string& p_name, const TypeImpl& p_type);

            // Default constructor to initialize members to empty values.
            Member();

            // The name of this member.
            std::string m_name;

            // The type of this member.
            const TypeImpl* m_type;
        };

        // Create a compound type of the give type, constness, and type 
        // manager.
        CompoundType(Type::TypePrimitive p_prim, bool p_isConst, TypeManager* p_typeManager);

        // Find a member by name. Returns a pointer to the member object 
        // associated with a name; if this compound type does not contain a 
        // member of the specified name, this function returns NULL.
        virtual const Member* FindMember(const std::string& p_name) const = 0;

        // Determine whether a TypePrimitive is a compound type.
        static bool IsCompoundType(const TypeImpl& p_type);
    };
}

#endif
