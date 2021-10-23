#include "Conditional.h"

#include "Expression.h"
#include "FreeForm2.h"
#include "FreeForm2Assert.h"
#include "SimpleExpressionOwner.h"
#include "Visitor.h"
#include <sstream>


FreeForm2::ConditionalExpression::ConditionalExpression(const Annotations& p_annotations,
                                                        const Expression& p_condition,
                                                        const Expression& p_then,
                                                        const Expression& p_else)
    : Expression(p_annotations),
      m_condition(p_condition), 
      m_then(p_then), 
      m_else(p_else)
{
}


const FreeForm2::Expression&
FreeForm2::ConditionalExpression::GetCondition() const
{
    return m_condition;
}


const FreeForm2::Expression&
FreeForm2::ConditionalExpression::GetThen() const
{
    return m_then;
}


const FreeForm2::Expression&
FreeForm2::ConditionalExpression::GetElse() const
{
    return m_else;
}


void
FreeForm2::ConditionalExpression::Accept(Visitor& p_visitor) const
{
    size_t stackSize = p_visitor.StackSize();

    if (!p_visitor.AlternativeVisit(*this))
    {
        m_else.Accept(p_visitor);
        m_then.Accept(p_visitor);
        m_condition.Accept(p_visitor);

        p_visitor.Visit(*this);
    }

    FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}


const FreeForm2::TypeImpl&
FreeForm2::ConditionalExpression::GetType() const
{
    if (m_condition.GetType().Primitive() != Type::Bool)
    {
        std::ostringstream err;
        err << "Condition of type '"
            << m_condition.GetType()
            << "' supplied to if expression as condition "
            << "(expected boolean)"; 
        throw ParseError(err.str(), GetSourceLocation());
    }

    if (!m_then.GetType().IsSameAs(m_else.GetType(), true))
    {
        if (m_then.GetType().IsIntegerType()
            && m_else.GetType().IsIntegerType())
        {
            return TypeImpl::GetIntInstance(true);
        }
        else
        {
            std::ostringstream err;
            err << "'then' (supplied '" 
                << m_then.GetType()
                << "' and 'else' (supplied '" 
                << m_else.GetType()
                << "' clauses of condition must have matching types.";
            throw ParseError(err.str(), GetSourceLocation());
        }
    }

    return m_then.GetType().AsConstType();
}


size_t
FreeForm2::ConditionalExpression::GetNumChildren() const
{
    return 3;
}

