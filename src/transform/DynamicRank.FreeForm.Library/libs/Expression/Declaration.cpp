#include "Declaration.h"

#include "ArrayType.h"
#include "FreeForm2Assert.h"
#include <sstream>
//#include <RankerFeatures.h>
#include "TypeUtil.h"
#include "Visitor.h"

FreeForm2::DeclarationExpression::DeclarationExpression(const Annotations& p_annotations,
                                                        const TypeImpl& p_type, 
                                                        const Expression& p_init,
                                                        bool p_voidValue,
                                                        VariableID p_id,
                                                        size_t p_version)
    : Expression(p_annotations),
      m_declType(p_type),
      m_init(p_init),
      m_voidValue(p_voidValue),
      m_id(p_id),
      m_version(p_version)
{
}


void 
FreeForm2::DeclarationExpression::Accept(Visitor& p_visitor) const
{
    size_t stackSize = p_visitor.StackSize();

    if (!p_visitor.AlternativeVisit(*this))
    {
        m_init.Accept(p_visitor);
        p_visitor.Visit(*this);
    }

    FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}


const FreeForm2::TypeImpl&
FreeForm2::DeclarationExpression::GetType() const
{
    if (m_declType.Primitive() != Type::Unknown && !TypeUtil::IsAssignable(m_declType, m_init.GetType()))
    {
        std::ostringstream err;
        err << "Declaration initializer (of type " 
            << m_init.GetType() 
            << ") does not match declared type of variable (" << m_declType << ")";
        throw ParseError(err.str(), GetSourceLocation());
    }

    if (m_voidValue)
    {
        return TypeImpl::GetVoidInstance();
    }
    else
    {
        FF2_ASSERT(m_init.GetType().Primitive() != Type::Unknown);
        FF2_ASSERT(m_init.GetType().Primitive() != Type::Void);
        return m_init.GetType().AsConstType();
    }
}


size_t 
FreeForm2::DeclarationExpression::GetNumChildren() const
{
    return 1;
}


const FreeForm2::Expression&
FreeForm2::DeclarationExpression::GetInit() const
{
    return m_init;
}


bool FreeForm2::DeclarationExpression::HasVoidValue() const
{
    return m_voidValue;
}


const FreeForm2::TypeImpl& 
FreeForm2::DeclarationExpression::GetDeclaredType() const
{
    return m_declType;
}


FreeForm2::VariableID
FreeForm2::DeclarationExpression::GetId() const
{
    return m_id;
}


size_t
FreeForm2::DeclarationExpression::GetVersion() const
{
    return m_version;
}
