#include "TypeUtil.h"

#include "ArrayType.h"
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include "ConvertExpression.h"
#include "FreeForm2Assert.h"
#include "LiteralExpression.h"
#include <sstream>
#include <stdexcept>
#include "StructType.h"
#include "TypeImpl.h"
#include "TypeManager.h"


namespace
{
    using namespace FreeForm2;

    template<typename F>
    boost::shared_ptr<FreeForm2::Expression>
    ConvertConstant(const Annotations& p_annotations, F p_value, Type::TypePrimitive p_to)
    {
        switch (p_to)
        {
            case Type::Int:
            {
                return boost::make_shared<LiteralIntExpression>(p_annotations, static_cast<Result::IntType>(p_value));
            }
            case Type::UInt64:
            {
                return boost::make_shared<LiteralUInt64Expression>(p_annotations, static_cast<Result::UInt64Type>(p_value));
            }
            case Type::Int32:
            {
                return boost::make_shared<LiteralInt32Expression>(p_annotations, static_cast<Result::Int32Type>(p_value));
            }
            case Type::UInt32:
            {
                return boost::make_shared<LiteralUInt32Expression>(p_annotations, static_cast<Result::UInt32Type>(p_value));
            }
            case Type::Float:
            {
                return boost::make_shared<LiteralFloatExpression>(p_annotations, static_cast<Result::FloatType>(p_value));
            }
            case Type::Bool:
            {
                return boost::make_shared<LiteralBoolExpression>(p_annotations, p_value != 0);
            }
            default:
            {
                throw std::bad_cast();
            }
        }
    }
}


bool
FreeForm2::TypeUtil::IsConvertible(const TypeImpl& p_from, const TypeImpl& p_to)
{
    if (p_from.Primitive() == Type::Unknown 
        || p_from.Primitive() == Type::Invalid
        || p_to.Primitive() == Type::Unknown 
        || p_to.Primitive() == Type::Invalid)
    {
        return false;
    }
    else if (p_from.Primitive() == Type::Array && p_to.Primitive() == Type::Array)
    {
        return false;
    }
    else if (p_from.IsLeafType() && p_to.IsLeafType())
    {
        return p_to.IsConst() || !p_from.IsConst();
    }
    else if (p_to.Primitive() == Type::Void)
    {
        return true;
    }

    return false;
}


boost::shared_ptr<FreeForm2::Expression>
FreeForm2::TypeUtil::Convert(const Expression& p_expr, Type::TypePrimitive p_type)
{
    if (p_type != Type::Void && p_expr.IsConstant())
    {
        return TypeUtil::ConvertConstant(p_expr.GetAnnotations(), p_expr.GetConstantValue(), p_expr.GetType().Primitive(), p_type);
    }

    switch (p_type)
    {
        case Type::Int:
        {
            return boost::make_shared<ConvertToIntExpression>(p_expr.GetAnnotations(), p_expr);
        }
        case Type::UInt64:
        {
            return boost::make_shared<ConvertToUInt64Expression>(p_expr.GetAnnotations(), p_expr);
        }
        case Type::Int32:
        {
            return boost::make_shared<ConvertToInt32Expression>(p_expr.GetAnnotations(), p_expr);
        }
        case Type::UInt32:
        {
            return boost::make_shared<ConvertToUInt32Expression>(p_expr.GetAnnotations(), p_expr);
        }
        case Type::Float:
        {
            return boost::make_shared<ConvertToFloatExpression>(p_expr.GetAnnotations(), p_expr);
        }
        case Type::Bool:
        {
            return boost::make_shared<ConvertToBoolExpression>(p_expr.GetAnnotations(), p_expr);
        }
        case Type::Void:
        {
            return boost::make_shared<ConvertToImperativeExpression>(p_expr.GetAnnotations(), p_expr);
        }
        default:
        {
            std::ostringstream err;
            err << "Unable to convert from " << p_expr.GetType()
                << " to " << Type::Name(p_type);
            throw ParseError(err.str(), p_expr.GetSourceLocation());
        }
    }
}


boost::shared_ptr<FreeForm2::Expression>
FreeForm2::TypeUtil::ConvertConstant(const Annotations& p_annotations, ConstantValue p_value, Type::TypePrimitive p_from, Type::TypePrimitive p_to)
{
    try
    {
        switch (p_from)
        {
            case Type::Int:
            {
                return ::ConvertConstant<Result::IntType>(p_annotations, p_value.m_int, p_to);
            }
            case Type::UInt64:
            {
                return ::ConvertConstant<Result::UInt64Type>(p_annotations, p_value.m_uint64, p_to);
            }
            case Type::Int32:
            {
                return ::ConvertConstant<Result::Int32Type>(p_annotations, p_value.m_int32, p_to);
            }
            case Type::UInt32:
            {
                return ::ConvertConstant<Result::UInt32Type>(p_annotations, p_value.m_uint32, p_to);
            }
            case Type::Float:
            {
                return ::ConvertConstant<Result::FloatType>(p_annotations, p_value.m_float, p_to);
            }
            case Type::Bool:
            {
                return ::ConvertConstant<Result::BoolType>(p_annotations, p_value.m_bool, p_to);
            }
            default:
            {
                throw std::bad_cast();
            }
        }
    }
    catch (std::bad_cast&)
    {
            std::ostringstream err;
            err << "Unable to convert from " << Type::Name(p_from)
                << " to " << Type::Name(p_to);
            throw ParseError(err.str(), p_annotations.m_sourceLocation);
    }
}


const FreeForm2::TypeImpl&
FreeForm2::TypeUtil::Unify(const TypeImpl& p_type1, 
                           const TypeImpl& p_type2, 
                           TypeManager& p_typeManager,
                           bool p_allowArray,
                           bool p_allowPromotion)
{
    if (!p_type1.IsValid() || !p_type2.IsValid())
    {
        return TypeImpl::GetInvalidType();
    }

    const bool isConst = p_type1.IsConst() || p_type2.IsConst();

    if (!p_allowArray && (p_type1.Primitive() == Type::Array 
        || p_type2.Primitive() == Type::Array))
    {
        std::ostringstream err;
        err << "We don't currently allow unification of array types, pending "
               "a decision on whether to track array bounds statically (TFS 62552).";
        throw std::runtime_error(err.str());
    }

    if (p_type1.IsSameAs(p_type2, true))
    {
        return p_type1.IsConst() ? p_type1 : p_type2;
    }

    if (p_type1.Primitive() == Type::Array && p_type2.Primitive() == Type::Array)
    {
        const ArrayType& left = static_cast<const ArrayType&>(p_type1);
        const ArrayType& right = static_cast<const ArrayType&>(p_type2);

        if (left.GetChildType().Primitive() == Type::Unknown 
            && left.GetDimensionCount() <= right.GetDimensionCount())
        {
            FF2_ASSERT(left.GetMaxElements() == 0);

            // Left is unknown, use right.
            return isConst ? p_type2.AsConstType() : p_type2;
        }
        else if (right.GetChildType().Primitive() == Type::Unknown
            && right.GetDimensionCount() <= left.GetDimensionCount())
        {
            FF2_ASSERT(right.GetMaxElements() == 0);

            // Right is unknown, use left.
            return isConst ? p_type1.AsConstType() : p_type1;
        }
        else if (left.GetDimensionCount() == right.GetDimensionCount()
            && left.GetChildType() == right.GetChildType())
        {
            // Return the type with more information. A fixed-size array 
            // provides the sizes of each dimension; we prefer to return the 
            // fixed-size array. Note that the case of two fix-sized arrays 
            // with the same dimensions is covered in the IsSameAs test.
            if (left.IsFixedSize() && !right.IsFixedSize())
            {
                return isConst ? left.AsConstType() : left;
            }
            else if (!left.IsFixedSize() && right.IsFixedSize())
            {
                return isConst ? right.AsConstType() : right;
            }
            // Otherwise, fall through to the error case.
        }
    }
    else if (p_type1.Primitive() == p_type2.Primitive())
    {
        return isConst ? p_type1.AsConstType() : p_type1;
    }
    else if (p_type1.Primitive() == Type::Unknown)
    {
        return isConst ? p_type2.AsConstType() : p_type2;
    }
    else if (p_type2.Primitive() == Type::Unknown)
    {
        return isConst ? p_type1.AsConstType() : p_type1;
    }
    else if (p_type1.IsLeafType() && p_type2.IsLeafType())
    {
        if (p_allowPromotion
            && (p_type1.Primitive() == Type::Float || p_type2.Primitive() == Type::Float)
            && (p_type1.IsIntegerType() || p_type2.IsIntegerType()))
        {
            return TypeImpl::GetFloatInstance(isConst);
        }
        
        // Note that no unification exists for uint64 and int64.
        if (p_allowPromotion
            && p_type1.IsIntegerType()
            && p_type2.IsIntegerType()
            && p_type1.Primitive() != Type::UInt64 
            && p_type2.Primitive() != Type::UInt64)
        {
            return TypeImpl::GetIntInstance(isConst);
        }

        if (p_allowPromotion
            && p_type1.IsIntegerType()
            && p_type2.IsIntegerType()
            && (p_type1.Primitive() == Type::UInt64 || p_type2.Primitive() == Type::UInt64)
            && p_type1.Primitive() != Type::Int 
            && p_type2.Primitive() != Type::Int)
        {
            return TypeImpl::GetUInt64Instance(isConst);
        }
    }
 
    return TypeImpl::GetInvalidType();
}


bool 
FreeForm2::TypeUtil::IsAssignable(const TypeImpl& p_dest, const TypeImpl& p_source)
{
    if (p_dest.Primitive() == Type::Array && p_source.Primitive() == Type::Array)
    {
        const ArrayType& source = static_cast<const ArrayType&>(p_source);
        const ArrayType& dest = static_cast<const ArrayType&>(p_dest);
        if (!source.GetChildType().IsSameAs(dest.GetChildType(), true))
        {
            return false;
        }

        if (source.GetDimensionCount() != dest.GetDimensionCount())
        {
            return false;
        }

        if (source.IsFixedSize() && dest.IsFixedSize())
        {
            if (memcmp(source.GetDimensions(), 
                            dest.GetDimensions(), 
                            sizeof(unsigned int) * source.GetDimensionCount()) != 0)
            {
                return false;
            }
            else
            {
                return true;
            }
        }
        else
        {
            return true;
        }
    }
    else
    {
        if (p_source.Primitive() == Type::Unknown || p_dest.Primitive() == Type::Unknown)
        {
            return true;
        }
        else if ((p_source.Primitive() == Type::Int32 || p_source.Primitive() == Type::UInt32)
                 && p_dest.Primitive() == Type::Int)
        {
            return true;
        }
        else if (p_source.Primitive() == Type::UInt32
                 && p_dest.Primitive() == Type::UInt64)
        {
            return true;
        }
        else
        {
            return p_source.IsSameAs(p_dest, true);
        }
    }
}



const FreeForm2::TypeImpl&
FreeForm2::TypeUtil::SetConstness(const TypeImpl& p_type, bool p_isConst)
{
    return p_isConst ? p_type.AsConstType() : p_type.AsMutableType();
}

