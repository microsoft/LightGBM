#include "Logic.h"

#include "BinaryOperator.h"
#include "OperatorExpressionFactory.h"
#include "UnaryOperator.h"

using namespace FreeForm2;

namespace
{
    typedef OperatorExpressionFactory OperatorFactory;
    static const OperatorFactory c_eqFactory(UnaryOperator::invalid, BinaryOperator::eq, false);
    static const OperatorFactory c_notEqFactory(UnaryOperator::invalid, BinaryOperator::neq, false);
    static const OperatorFactory c_ltFactory(UnaryOperator::invalid, BinaryOperator::lt, false);
    static const OperatorFactory c_lteFactory(UnaryOperator::invalid, BinaryOperator::lte, false);
    static const OperatorFactory c_gtFactory(UnaryOperator::invalid, BinaryOperator::gt, false);
    static const OperatorFactory c_gteFactory(UnaryOperator::invalid, BinaryOperator::gte, false);
    static const OperatorFactory c_andFactory(UnaryOperator::invalid, BinaryOperator::_and, true);
    static const OperatorFactory c_orFactory(UnaryOperator::invalid, BinaryOperator::_or, true);
    static const OperatorFactory c_notFactory(UnaryOperator::_not, BinaryOperator::invalid, false);
}



const FreeForm2::ExpressionFactory& 
FreeForm2::Logic::GetCmpEqInstance()
{
    return c_eqFactory;
}


const FreeForm2::ExpressionFactory& 
FreeForm2::Logic::GetCmpNotEqInstance()
{
    return c_notEqFactory;
}


const FreeForm2::ExpressionFactory& 
FreeForm2::Logic::GetCmpLTInstance()
{
    return c_ltFactory;
}


const FreeForm2::ExpressionFactory& 
FreeForm2::Logic::GetCmpLTEInstance()
{
    return c_lteFactory;
}


const FreeForm2::ExpressionFactory& 
FreeForm2::Logic::GetCmpGTInstance()
{
    return c_gtFactory;
}


const FreeForm2::ExpressionFactory& 
FreeForm2::Logic::GetCmpGTEInstance()
{
    return c_gteFactory;
}


const FreeForm2::ExpressionFactory& 
FreeForm2::Logic::GetAndInstance()
{
    return c_andFactory;
}


const FreeForm2::ExpressionFactory& 
FreeForm2::Logic::GetOrInstance()
{
    return c_orFactory;
}


const FreeForm2::ExpressionFactory& 
FreeForm2::Logic::GetNotInstance()
{
    return c_notFactory;
}

