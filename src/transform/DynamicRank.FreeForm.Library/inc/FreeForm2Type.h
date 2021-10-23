#pragma once

#ifndef FREEFORM2_TYPE_H
#define FREEFORM2_TYPE_H

#include <basic_types.h>
#include <boost/operators.hpp>
#include <boost/noncopyable.hpp>
#include <boost/scoped_ptr.hpp>
#include <iosfwd>
#include <memory>

namespace FreeForm2
{
    class TypeImpl;
    class TypeManager;

    class Type : public boost::equality_comparable<Type>
    {
    public:
        // List of possible value types.  Note that i'm not following the coding
        // guidelines here (by capitalising each), because otherwise we get
        // conflicts with the C++ type 'float', etc.
        enum TypePrimitive
        {
            Float,
            Int,
            UInt64,
            Int32,
            UInt32,
            Bool,
            Array,
            Struct,
            Void,
            Stream,
            Word,
            InstanceHeader,
            BodyBlockHeader,
            StateMachine,
            Function,
            Object,

            Unknown,
            Invalid
        };

        // Construct a type from a TypeImpl.  Note that no ownership transfer is
        // implied by this, and the TypeImpl object must remain in scope for the
        // lifetime of the Type.
        explicit Type(const TypeImpl& p_impl);

        // Get the type primitive of this type.
        TypePrimitive Primitive() const;

        // Returns a name for each type in the Type enumeration, or NULL for
        // unrecognised types.
        static const char* Name(TypePrimitive p_type);

        // Returns the type primitive corresponding to a name, the inverse of
        // the Name() function.  Returns Type::Invalid for unrecognised names,
        // and is case sensitive.
        static TypePrimitive ParsePrimitive(SIZED_STRING p_string);

        // Equality operator.
        bool operator==(const Type& p_other) const;

        // Get implementation class.
        const TypeImpl& GetImplementation() const;

    private:
        Type(const Type& p_type);
        void operator=(const Type& p_type);

        // Pointer to implementation (pimpl idiom).
        const TypeImpl& m_impl;
    };

    // Output to std::ostream.
    std::ostream& operator<<(std::ostream& p_out, const Type& p_type);

    // A type factory allows creation of arbitrary TypeImpl objects without
    // exposing the type implementations.
    class TypeFactory : boost::noncopyable
    {
    public:
        // Construct a TypeFactory for an implementation.
        TypeFactory(std::auto_ptr<TypeManager> p_typeManager);

        // The members of a structure consists of a name-type pair.
        typedef std::pair<std::string, const TypeImpl&> StructMember;

        // These methods produce a reference to a TypeImpl corresponding to a
        // feature compiler type. These can be used to construct Type objects.
        static const TypeImpl& GetFloatType();
        static const TypeImpl& GetIntType();
        static const TypeImpl& GetUInt64Type();
        static const TypeImpl& GetInt32Type();
        static const TypeImpl& GetUInt32Type();
        static const TypeImpl& GetBoolType();
        static const TypeImpl& GetVoidType();
        static const TypeImpl& GetStreamType();
        static const TypeImpl& GetWordType();
        static const TypeImpl& GetInstanceHeaderType();
        static const TypeImpl& GetBodyBlockHeaderType();
        const TypeImpl& GetArrayType(const TypeImpl& p_child, 
                                     const UInt32* p_dimensions,
                                     UInt32 p_dimensionCount);
        const TypeImpl& GetStructType(const std::string& p_name,
                                      const StructMember* p_members,
                                      UInt32 p_memberCount);
        const TypeImpl& GetFunctionType(const TypeImpl& p_returnType,
                                        const TypeImpl* const* p_parameters,
                                        UInt32 p_parameterCount);

        // Find a type by type name. This should be the same string as when the
        // type is written to a stream using the stream insertion operator.
        const TypeImpl* FindType(const std::string& p_name) const;

        // Get the type manager associated with this factory.
        const TypeManager& GetTypeManager() const;

    private:
        // The type manager owns the types created with the above methods.
        boost::scoped_ptr<TypeManager> m_typeManager;
    };
}

#endif
