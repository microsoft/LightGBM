#pragma once

#ifndef FREEFORM2_TYPE_UTIL_H
#define FREEFORM2_TYPE_UTIL_H

#include <boost/shared_ptr.hpp>
#include "Expression.h"
#include "TypeImpl.h"

namespace FreeForm2
{
    class ArrayType;
    class ConversionExpression; 
    class TypeManager;

    class TypeUtil
    {
    public:
        // Check if one type is convertible to another type. Returns true if
        // a ConversionExpression exists; false otherwise.
        static bool IsConvertible(const TypeImpl& p_from, const TypeImpl& p_to);

        // Create a new Expression of the specified type for a child 
        // expression.
        static boost::shared_ptr<Expression> Convert(const Expression& p_expr,
                                                     Type::TypePrimitive p_type);

        // Create a new literal Expression of the specified type for a constant
        // value.
        static boost::shared_ptr<Expression> ConvertConstant(const Annotations& p_annotations,
                                                             ConstantValue p_value,
                                                             Type::TypePrimitive p_from,
                                                             Type::TypePrimitive p_to);

        // Return a type that's compatible with two types, or return 
        // Type::Invalid. If the types differ in const-ness, the resulting
        // type will be constant. p_allowArray specifies whether we are 
        // allowed to unify array types.
        static const TypeImpl& Unify(const TypeImpl& p_type1, 
                                     const TypeImpl& p_type2, 
                                     TypeManager& p_typeManager,
                                     bool p_allowArray,
                                     bool p_allowPromotion);

        // Determine if a source type is assignable to a destination type.
        static bool IsAssignable(const TypeImpl& p_dest, 
                                 const TypeImpl& p_source);

        // Return a type which matches the type argument, but with const-ness
        // specified by a flag.
        static const TypeImpl& SetConstness(const TypeImpl& p_type, bool p_isConst);
    };
}

#endif
