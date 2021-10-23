#include "ArrayDereferenceExpression.h"

#include "ArrayType.h"
#include "FreeForm2Assert.h"
#include "RefExpression.h"
#include "SimpleExpressionOwner.h"
#include <sstream>
#include "TypeManager.h"
#include "TypeUtil.h"
#include "Visitor.h"

namespace
{
    const FreeForm2::TypeImpl& 
    DerefType(const FreeForm2::TypeImpl& p_arrayType, const FreeForm2::SourceLocation& p_sourceLocation)
    {
        if (p_arrayType.Primitive() != FreeForm2::Type::Array)
        {
            std::ostringstream err;
            err << "The array operand in an array dereference expression "
                << "is not an array (instead, it is a " 
                << p_arrayType << ")";
            throw FreeForm2::ParseError(err.str(), p_sourceLocation);
        }

        const FreeForm2::ArrayType& arrayType = static_cast<const FreeForm2::ArrayType&>(p_arrayType);
        return arrayType.GetDerefType();
    }
}


FreeForm2::ArrayDereferenceExpression::ArrayDereferenceExpression(const Annotations& p_annotations,
                                                                  const Expression& p_array, 
                                                                  const Expression& p_index,
                                                                  size_t p_version)
    : Expression(p_annotations),
      m_type(DerefType(p_array.GetType(), p_annotations.m_sourceLocation)),
      m_array(p_array), 
      m_index(p_index),
      m_version(p_version)
{
}


const FreeForm2::TypeImpl&
FreeForm2::ArrayDereferenceExpression::GetType() const
{
    if (!m_index.GetType().IsIntegerType())
    {
        std::ostringstream err;
        err << "The index operand in an array dereference expression "
            << "is not an integer type (instead, it is a " 
            << m_index.GetType() << ")";
        throw ParseError(err.str(), GetSourceLocation());
    }

    return m_type;
}


size_t
FreeForm2::ArrayDereferenceExpression::GetNumChildren() const
{
    return 2;
}


void
FreeForm2::ArrayDereferenceExpression::Accept(Visitor& p_visitor) const
{
    size_t stackSize = p_visitor.StackSize();

    if (!p_visitor.AlternativeVisit(*this))
    {
        m_array.Accept(p_visitor);
        m_index.Accept(p_visitor);

        p_visitor.Visit(*this);
    }

    FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}


void
FreeForm2::ArrayDereferenceExpression::AcceptReference(Visitor& p_visitor) const
{
    size_t stackSize = p_visitor.StackSize();

    if (!p_visitor.AlternativeVisitReference(*this))
    {
        m_array.AcceptReference(p_visitor);
        m_index.Accept(p_visitor);

        p_visitor.VisitReference(*this);
    }

    FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}


const FreeForm2::Expression&
FreeForm2::ArrayDereferenceExpression::GetArray() const
{
    return m_array;
}


const FreeForm2::Expression&
FreeForm2::ArrayDereferenceExpression::GetIndex() const
{
    return m_index;
}


size_t
FreeForm2::ArrayDereferenceExpression::GetVersion() const
{
    return m_version;
}


FreeForm2::VariableID
FreeForm2::ArrayDereferenceExpression::GetBaseArrayId() const
{
    const ArrayDereferenceExpression* deref = this;
    const Expression* array = &GetArray();

    while (deref != nullptr)
    {
        array = &deref->GetArray();
        deref = dynamic_cast<const ArrayDereferenceExpression*>(array);
    }

    const VariableRefExpression* base
        = dynamic_cast<const VariableRefExpression*>(array);

    if (base != nullptr)
    {
        return base->GetId();
    }
    else
    {
        // The base array is a literal.
        return VariableID::c_invalidID;
    }
}
