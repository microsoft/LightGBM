#pragma once

#ifndef FREEFORM2_TYPEIMPL_H
#define FREEFORM2_TYPEIMPL_H

#include <boost/noncopyable.hpp>
#include "FreeForm2Type.h"
#include <ostream>
#include <string>
#include <vector>

namespace FreeForm2 
{
    class TypeManager;

    class TypeImpl : boost::noncopyable
    {
    public:
        // Construct a type implementation from the given type primitive 
        // and const-ness.
        TypeImpl(Type::TypePrimitive p_prim, bool p_isConst, TypeManager* p_typeManager);

        // Get the type primitive of this type.
        Type::TypePrimitive Primitive() const;

        // Equality comparison for types. This function returns true if two 
        // types are equivalent; otherwise returns false. This function will
        // ignore constness if p_ignoreConst is true.
        bool IsSameAs(const TypeImpl& p_type, bool p_ignoreConst) const;

        // Equality operator. Implemented in terms of IsSameAs, taking 
        // constness into account.
        bool operator==(const TypeImpl& p_other) const;

        // Inequality operator. Implemented in terms of operator ==.
        bool operator!=(const TypeImpl& p_other) const;

        // Get a string representation of the type.
        virtual const std::string& GetName() const = 0;

        // A static and member function to determine whether a given or the
        // current type is of fixed size.  Types that are not of fixed size
        // include arrays (variable-sized) and void (doesn't have a size).
        static bool IsLeafType(Type::TypePrimitive p_prim);
        bool IsLeafType() const;

        // Check if this type is an integer type.
        bool IsIntegerType() const;

        // Check if this type is a floating-point type.
        bool IsFloatingPointType() const;

        // Check if this type is signed. Signed types include signed integer
        // types and the float type.
        bool IsSigned() const;

        // Check if this type is valid. Unknown types are valid.
        bool IsValid() const;

        // Check if this type is const.
        bool IsConst() const;

        // Create derived types based on this type.
        virtual const TypeImpl& AsConstType() const = 0;
        virtual const TypeImpl& AsMutableType() const = 0;

        // Return convenience single instances of some common, leaf types.
        static const TypeImpl& GetFloatInstance(bool p_isConst);
        static const TypeImpl& GetIntInstance(bool p_isConst);
        static const TypeImpl& GetUInt64Instance(bool p_isConst);
        static const TypeImpl& GetInt32Instance(bool p_isConst);
        static const TypeImpl& GetUInt32Instance(bool p_isConst);
        static const TypeImpl& GetBoolInstance(bool p_isConst);
        static const TypeImpl& GetVoidInstance();
        static const TypeImpl& GetStreamInstance(bool p_isConst);
        static const TypeImpl& GetWordInstance(bool p_isConst);
        static const TypeImpl& GetInstanceHeaderInstance(bool p_isConst);
        static const TypeImpl& GetBodyBlockHeaderInstance(bool p_isConst);
        static const TypeImpl& GetUnknownType();
        static const TypeImpl& GetInvalidType();

        // Get the type for the singleton types above.
        static const TypeImpl& GetCommonType(Type::TypePrimitive p_prim, bool p_isConst);

    protected:
        // For derived classes, return the TypeManager passed to the 
        // constructor of this TypeImpl.
        TypeManager* GetTypeManager() const;

    private:
        // This method determines if the subclass data for this TypeImpl is the
        // same as the paramter. This method may assume that Primitive() ==
        // p_type.Primitive().
        virtual bool IsSameSubType(const TypeImpl& p_type, bool p_ignoreConst) const = 0;

        // Type primitive of this type.
        Type::TypePrimitive m_prim;

        // Constness of the type.
        bool m_isConst;

        // Owning TypeManager of this type.
        TypeManager* m_typeManager;
    };

    std::ostream&
    operator<<(std::ostream& p_out, const TypeImpl& p_type);
}

#endif

