#include "FunctionType.h"

#include <boost/foreach.hpp>
#include "FreeForm2Assert.h"
#include <sstream>
#include "TypeManager.h"


FreeForm2::FunctionType::FunctionType(TypeManager& p_typeManager,
                                      const TypeImpl& p_returnType,
                                      const TypeImpl* const* p_parameterTypes,
                                      size_t p_numParameters)
    : TypeImpl(Type::Function, true, &p_typeManager),
      m_returnType(&p_returnType),
      m_numParameters(p_numParameters)
{
    if (m_numParameters > 0)
    {
        m_parameterTypes[0] = p_parameterTypes[0];

        for (size_t i = 1; i < m_numParameters; i++)
        {
            m_parameterTypes[i] = p_parameterTypes[i];
        }
    }
}


const FreeForm2::TypeImpl&
FreeForm2::FunctionType::GetReturnType() const
{
    return *m_returnType;
}


FreeForm2::FunctionType::ParameterIterator
FreeForm2::FunctionType::BeginParameters() const
{
    return const_cast<FreeForm2::FunctionType::ParameterIterator>(&m_parameterTypes[0]);
}


FreeForm2::FunctionType::ParameterIterator
FreeForm2::FunctionType::EndParameters() const
{
    return BeginParameters() + m_numParameters;
}


size_t
FreeForm2::FunctionType::GetParameterCount() const
{
    return m_numParameters;
}


const std::string&
FreeForm2::FunctionType::GetName() const
{
    if (m_name.size() == 0)
    {
        m_name = FunctionType::GetName(GetReturnType(), BeginParameters(), GetParameterCount());
    }

    return m_name;
}


std::string
FreeForm2::FunctionType::GetName(const TypeImpl& p_returnType,
                                 const TypeImpl* const* p_parameterTypes,
                                 size_t p_numParams)
{
    std::ostringstream out;
    out << p_returnType << "(";

    bool first = true;

    for (int i = 0; i < p_numParams; i++)
    {
        if (!first)
        {
            out << ", ";
        }

        out << *p_parameterTypes[i];

        first = false;
    }

    out << ")";

    return out.str();
}


const FreeForm2::TypeImpl& 
FreeForm2::FunctionType::AsConstType() const
{
    return *this;
}


const FreeForm2::TypeImpl& 
FreeForm2::FunctionType::AsMutableType() const
{
    FF2_ASSERT("Functions cannot be mutable." && false);
    Unreachable(__FILE__, __LINE__);
}


bool
FreeForm2::FunctionType::IsSameSubType(const TypeImpl& p_other, bool p_ignoreConst) const
{
    FF2_ASSERT(p_other.Primitive() == Type::Function);
    const FunctionType& other = static_cast<const FunctionType&>(p_other);

    if (m_numParameters != other.m_numParameters)
    {
        return false;
    }

    for (size_t i = 0; i < m_numParameters; i++)
    {
        const TypeImpl& m1 = *m_parameterTypes[i];
        const TypeImpl& m2 = *other.m_parameterTypes[i];
        if (!m1.IsSameAs(m2, p_ignoreConst))
        {
            return false;
        }
    }

    return GetReturnType().IsSameAs(other.GetReturnType(), p_ignoreConst);
}
