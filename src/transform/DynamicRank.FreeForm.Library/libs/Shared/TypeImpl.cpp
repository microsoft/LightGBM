#include "TypeImpl.h"

#include "ArrayType.h"
#include "FreeForm2Assert.h"
#include <sstream>
#include "StructType.h"

namespace
{
    std::string GetTypeName(FreeForm2::Type::TypePrimitive p_type, bool p_isConst)
    {
        std::ostringstream out;
        if (!p_isConst)
        {
            out << "mutable ";
        }
        out << FreeForm2::Type::Name(p_type);
        return out.str();
    }

    // A class representing the builtin primitive types.
    class PrimitiveType : public FreeForm2::TypeImpl
    {
    public:
        PrimitiveType(FreeForm2::Type::TypePrimitive p_prim, bool p_isConst)
            : TypeImpl(p_prim, p_isConst, NULL), m_name(GetTypeName(p_prim, p_isConst))
        {
        }

        virtual const std::string&
        GetName() const override
        {
            return m_name;
        }

        virtual const TypeImpl& AsConstType() const override
        {
            using FreeForm2::Type;
            switch (Primitive())
            {
                case Type::Float: return TypeImpl::GetFloatInstance(true);
                case Type::Int: return TypeImpl::GetIntInstance(true);
                case Type::UInt64: return TypeImpl::GetUInt64Instance(true);
                case Type::Int32: return TypeImpl::GetInt32Instance(true);
                case Type::UInt32: return TypeImpl::GetUInt32Instance(true);
                case Type::Bool: return TypeImpl::GetBoolInstance(true);
                case Type::Stream: return TypeImpl::GetStreamInstance(true);
                case Type::Word: return TypeImpl::GetWordInstance(true);
                case Type::InstanceHeader: return TypeImpl::GetInstanceHeaderInstance(true);
                case Type::BodyBlockHeader: return TypeImpl::GetBodyBlockHeaderInstance(true);

                // For Void, Unknown, and Invalid, constness is not definable.
                case Type::Void: __attribute__((__fallthrough__));
                case Type::Unknown: __attribute__((__fallthrough__));
                case Type::Invalid: return *this;

                default: FreeForm2::Unreachable(__FILE__, __LINE__);
            }
        }

        virtual const TypeImpl& AsMutableType() const override
        {
            using FreeForm2::Type;
            switch (Primitive())
            {
                case Type::Float: return TypeImpl::GetFloatInstance(false);
                case Type::Int: return TypeImpl::GetIntInstance(false);
                case Type::UInt64: return TypeImpl::GetUInt64Instance(false);
                case Type::Int32: return TypeImpl::GetInt32Instance(false);
                case Type::UInt32: return TypeImpl::GetUInt32Instance(false);
                case Type::Bool: return TypeImpl::GetBoolInstance(false);
                case Type::Stream: return TypeImpl::GetStreamInstance(false);
                case Type::Word: return TypeImpl::GetWordInstance(false);
                case Type::InstanceHeader: return TypeImpl::GetInstanceHeaderInstance(false);
                case Type::BodyBlockHeader: return TypeImpl::GetBodyBlockHeaderInstance(false);

                // For Void, Unknown, and Invalid, constness is not definable.
                case Type::Void: __attribute__((__fallthrough__));
                case Type::Unknown: __attribute__((__fallthrough__));
                case Type::Invalid: return *this;

                default: FreeForm2::Unreachable(__FILE__, __LINE__);
            }
        }

    private:
        virtual bool IsSameSubType(const TypeImpl& p_other, bool p_ignoreConst) const override
        {
            FF2_ASSERT(Primitive() == p_other.Primitive());
            return true;
        }

        std::string m_name;
    };
}


FreeForm2::TypeImpl::TypeImpl(Type::TypePrimitive p_prim, bool p_isConst, TypeManager* p_typeManager)
    : m_prim(p_prim),
      m_isConst(p_isConst),
      m_typeManager(p_typeManager)
{
}


FreeForm2::Type::TypePrimitive 
FreeForm2::TypeImpl::Primitive() const
{
    return m_prim;
}


bool
FreeForm2::TypeImpl::IsSameAs(const TypeImpl& p_other, bool p_ignoreConst) const
{
    return (Primitive() == p_other.Primitive()
            && (p_ignoreConst || IsConst() == p_other.IsConst())
            && IsSameSubType(p_other, p_ignoreConst));
}


bool
FreeForm2::TypeImpl::operator==(const TypeImpl& p_other) const
{
    return IsSameAs(p_other, false);
}


bool 
FreeForm2::TypeImpl::operator!=(const TypeImpl& p_other) const
{
    return !(*this == p_other);
}


bool 
FreeForm2::TypeImpl::IsLeafType(Type::TypePrimitive p_prim)
{
    switch (p_prim)
    {
        case Type::Bool:
        case Type::Int:
        case Type::UInt64:
        case Type::Int32:
        case Type::UInt32:
        case Type::Float:
        {
            return true;
        }

        // Note that Type::Void does not have fixed size (it effectively has no
        // size).
        case Type::Void:
        case Type::Unknown:
        case Type::Array:
        case Type::Struct:
        case Type::Stream:
        case Type::Word:
        case Type::InstanceHeader:
        case Type::BodyBlockHeader:
        case Type::StateMachine:
        case Type::Function:
        case Type::Object:
        {
            return false;
        }

        default:
        {
            Unreachable(__FILE__, __LINE__);
            break;
        }
    };
}


bool 
FreeForm2::TypeImpl::IsLeafType() const
{
    return IsLeafType(m_prim);
}


bool 
FreeForm2::TypeImpl::IsIntegerType() const
{
    return (m_prim == Type::Int)
           || (m_prim == Type::UInt64)
           || (m_prim == Type::Int32)
           || (m_prim == Type::UInt32);
}


bool
FreeForm2::TypeImpl::IsFloatingPointType() const
{
    return m_prim == Type::Float;
}


bool
FreeForm2::TypeImpl::IsSigned() const
{
    return (m_prim == Type::Int)
            || (m_prim == Type::Int32)
            || (m_prim == Type::Float);
}


bool
FreeForm2::TypeImpl::IsValid() const
{
    return Primitive() != Type::Invalid;
}


bool
FreeForm2::TypeImpl::IsConst() const
{
    return m_isConst;
}


FreeForm2::TypeManager*
FreeForm2::TypeImpl::GetTypeManager() const
{
    return m_typeManager;
}


std::ostream& 
FreeForm2::operator<<(std::ostream& p_out, const TypeImpl& p_type)
{
    return p_out << p_type.GetName();
}


const FreeForm2::TypeImpl& 
FreeForm2::TypeImpl::GetFloatInstance(bool p_isConst)
{
    static const PrimitiveType type(Type::Float, false);
    static const PrimitiveType constType(Type::Float, true);
    return p_isConst ? constType : type;
}


const FreeForm2::TypeImpl& 
FreeForm2::TypeImpl::GetIntInstance(bool p_isConst)
{
    static const PrimitiveType type(Type::Int, false);
    static const PrimitiveType constType(Type::Int, true);
    return p_isConst ? constType : type;
}


const FreeForm2::TypeImpl& 
FreeForm2::TypeImpl::GetUInt64Instance(bool p_isConst)
{
    static const PrimitiveType type(Type::UInt64, false);
    static const PrimitiveType constType(Type::UInt64, true);
    return p_isConst ? constType : type;
}


const FreeForm2::TypeImpl& 
FreeForm2::TypeImpl::GetInt32Instance(bool p_isConst)
{
    static const PrimitiveType type(Type::Int32, false);
    static const PrimitiveType constType(Type::Int32, true);
    return p_isConst ? constType : type;
}


const FreeForm2::TypeImpl& 
FreeForm2::TypeImpl::GetUInt32Instance(bool p_isConst)
{
    static const PrimitiveType type(Type::UInt32, false);
    static const PrimitiveType constType(Type::UInt32, true);
    return p_isConst ? constType : type;
}


const FreeForm2::TypeImpl& 
FreeForm2::TypeImpl::GetBoolInstance(bool p_isConst)
{
    static const PrimitiveType type(Type::Bool, false);
    static const PrimitiveType constType(Type::Bool, true);
    return p_isConst ? constType : type;
}


const FreeForm2::TypeImpl& 
FreeForm2::TypeImpl::GetVoidInstance()
{
    static const PrimitiveType type(Type::Void, true);
    return type;
}


const FreeForm2::TypeImpl& 
FreeForm2::TypeImpl::GetWordInstance(bool p_isConst)
{
    static const PrimitiveType type(Type::Word, false);
    static const PrimitiveType constType(Type::Word, true);
    return p_isConst ? constType : type;
}


const FreeForm2::TypeImpl& 
FreeForm2::TypeImpl::GetInstanceHeaderInstance(bool p_isConst)
{
    static const PrimitiveType type(Type::InstanceHeader, false);
    static const PrimitiveType constType(Type::InstanceHeader, true);
    return p_isConst ? constType : type;
}


const FreeForm2::TypeImpl& 
FreeForm2::TypeImpl::GetBodyBlockHeaderInstance(bool p_isConst)
{
    static const PrimitiveType type(Type::BodyBlockHeader, false);
    static const PrimitiveType constType(Type::BodyBlockHeader, true);
    return p_isConst ? constType : type;
}


const FreeForm2::TypeImpl& 
FreeForm2::TypeImpl::GetStreamInstance(bool p_isConst)
{
    static const PrimitiveType type(Type::Stream, false);
    static const PrimitiveType constType(Type::Stream, true);
    return p_isConst ? constType : type;
}


const FreeForm2::TypeImpl& 
FreeForm2::TypeImpl::GetUnknownType()
{
    static const PrimitiveType type(Type::Unknown, true);
    return type;
}


const FreeForm2::TypeImpl& 
FreeForm2::TypeImpl::GetInvalidType()
{
    static const PrimitiveType type(Type::Invalid, true);
    return type;
}


const FreeForm2::TypeImpl& 
FreeForm2::TypeImpl::GetCommonType(Type::TypePrimitive p_prim, bool p_isConst)
{
    switch (p_prim)
    {
        case Type::Float: return TypeImpl::GetFloatInstance(p_isConst);
        case Type::Int: return TypeImpl::GetIntInstance(p_isConst);
        case Type::UInt64: return TypeImpl::GetUInt64Instance(p_isConst);
        case Type::Int32: return TypeImpl::GetInt32Instance(p_isConst);
        case Type::UInt32: return TypeImpl::GetUInt32Instance(p_isConst);
        case Type::Bool: return TypeImpl::GetBoolInstance(p_isConst);
        case Type::Void: return TypeImpl::GetVoidInstance();
        case Type::Stream: return TypeImpl::GetStreamInstance(p_isConst);
        case Type::Word: return TypeImpl::GetWordInstance(p_isConst);
        case Type::InstanceHeader: return TypeImpl::GetInstanceHeaderInstance(p_isConst);
        case Type::BodyBlockHeader: return TypeImpl::GetBodyBlockHeaderInstance(p_isConst);
        case Type::Unknown: return TypeImpl::GetUnknownType();
        case Type::Invalid: return TypeImpl::GetInvalidType();
        default: Unreachable(__FILE__, __LINE__);
    }
}

