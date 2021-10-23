#include "Bitwise.h"

#include "BinaryOperator.h"
#include "OperatorExpressionFactory.h"
#include "UnaryOperator.h"

using namespace FreeForm2;

namespace
{
    typedef OperatorExpressionFactory OperatorExpression;
    static const OperatorExpression c_and(UnaryOperator::invalid, BinaryOperator::_bitand, false);
    static const OperatorExpression c_or(UnaryOperator::invalid, BinaryOperator::_bitor, false);
    static const OperatorExpression c_not(UnaryOperator::bitnot, BinaryOperator::invalid, false);
}


const FreeForm2::ExpressionFactory& 
FreeForm2::Bitwise::GetAndInstance()
{
    return c_and;
}


const FreeForm2::ExpressionFactory& 
FreeForm2::Bitwise::GetOrInstance()
{
    return c_or;
}


const FreeForm2::ExpressionFactory& 
FreeForm2::Bitwise::GetNotInstance()
{
    return c_not;
}

