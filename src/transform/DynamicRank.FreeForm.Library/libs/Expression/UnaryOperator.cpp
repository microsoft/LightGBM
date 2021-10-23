#include "UnaryOperator.h"

#include "FreeForm2Assert.h"
#include "FreeForm2Type.h"
#include "TypeImpl.h"
#include "TypeUtil.h"
#include <vector>

static
std::vector<FreeForm2::Type::TypePrimitive>
GetOperandTypes(FreeForm2::UnaryOperator::Operation p_op)
{
    using FreeForm2::UnaryOperator;
    std::vector<FreeForm2::Type::TypePrimitive> types;
    types.reserve(2);

    switch (p_op)
    {
    case UnaryOperator::_not:
        types.push_back(FreeForm2::Type::Bool);
        break;

    case UnaryOperator::abs: __attribute__((__fallthrough__));
    case UnaryOperator::minus: __attribute__((__fallthrough__));
    case UnaryOperator::round: __attribute__((__fallthrough__));
    case UnaryOperator::trunc: __attribute__((__fallthrough__));
    case UnaryOperator::log: __attribute__((__fallthrough__));
    case UnaryOperator::log1: __attribute__((__fallthrough__));
    case UnaryOperator::tanh:
        types.push_back(FreeForm2::Type::Int);
        types.push_back(FreeForm2::Type::Int32);
        types.push_back(FreeForm2::Type::UInt32);
        types.push_back(FreeForm2::Type::UInt64);
        types.push_back(FreeForm2::Type::Float);
        break;

    case UnaryOperator::bitnot:
        types.push_back(FreeForm2::Type::Int);
        types.push_back(FreeForm2::Type::Int32);
        types.push_back(FreeForm2::Type::UInt32);
        types.push_back(FreeForm2::Type::UInt64);
        break;

    default:
        FreeForm2::Unreachable(__FILE__, __LINE__);
    }

    return types;
}

const FreeForm2::TypeImpl&
FreeForm2::UnaryOperator::GetBestOperandType(Operation p_operator, 
                                             const TypeImpl& p_operand)
{
    if (p_operand.Primitive() == Type::Unknown)
    {
        return p_operand;
    }

    const std::vector<Type::TypePrimitive> types = GetOperandTypes(p_operator);

    if (std::find(types.begin(), types.end(), p_operand.Primitive()) != types.end())
    {
        return p_operand;
    }
    else
    {
        return TypeImpl::GetInvalidType();
    }
}

const FreeForm2::TypeImpl&
FreeForm2::UnaryOperator::GetReturnType(Operation p_operator, 
                                        const TypeImpl& p_operand)
{
    switch (p_operator)
    {
        case UnaryOperator::minus:
        {
            if (p_operand.Primitive() == Type::UInt32)
            {
                return TypeImpl::GetIntInstance(true);
            }
            else
            {
                return p_operand;
            }
        }
        case UnaryOperator::trunc: __attribute__((__fallthrough__));
        case UnaryOperator::round:
        {
            return TypeImpl::GetIntInstance(true);
        }
        case UnaryOperator::log: __attribute__((__fallthrough__));
        case UnaryOperator::log1: __attribute__((__fallthrough__)); 
        case UnaryOperator::tanh:
        {
            return TypeImpl::GetFloatInstance(true);
        }
        default:
        {
            return p_operand;
        }
    }
}
