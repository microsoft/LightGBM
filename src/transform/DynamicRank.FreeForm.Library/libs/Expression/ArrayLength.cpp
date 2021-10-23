#include "ArrayLength.h"

#include "FreeForm2Assert.h"
#include "Expression.h"
#include "SimpleExpressionOwner.h"
#include "Visitor.h"
#include <sstream>

FreeForm2::ArrayLengthExpression::ArrayLengthExpression(const Annotations& p_annotations,
                                                        const Expression& p_array)
    : Expression(p_annotations),
      m_array(p_array)
{
}


void 
FreeForm2::ArrayLengthExpression::Accept(Visitor& p_visitor) const
{
    size_t stackSize = p_visitor.StackSize();

    if (!p_visitor.AlternativeVisit(*this))
    {
        m_array.Accept(p_visitor);

        p_visitor.Visit(*this);
    }

    FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}


const FreeForm2::TypeImpl&
FreeForm2::ArrayLengthExpression::GetType() const
{
    if (m_array.GetType().Primitive() != Type::Array)
    {
        std::ostringstream err;
        err << "Argument to array-length expression must be "
            << "an array (got type '"
            << m_array.GetType() << "')";
        throw ParseError(err.str(), GetSourceLocation());
    }

    return TypeImpl::GetUInt32Instance(true);
}


size_t
FreeForm2::ArrayLengthExpression::GetNumChildren() const
{
    return 1;
}

