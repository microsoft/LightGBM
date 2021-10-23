#pragma once

#ifndef FREEFORM2_BINARY_OPERATOR_H
#define FREEFORM2_BINARY_OPERATOR_H

#include "Expression.h"
#include "FreeForm2Type.h"

namespace FreeForm2
{
    class BinaryOperator
    {
    public:
        enum Operation
        {
            plus,
            minus,
            multiply,
            divides,
            mod,
            max,
            min,
            pow,
            log,

            eq,
            neq,
            lt,
            lte,
            gt,
            gte,

            _and,
            _or,

            _bitand,
            _bitor,
            bitshiftleft,
            bitshiftright,

            invalid
        };

        // Select the best operand type for an operator. Best is defined in 
        // terms of TypeUtil::SelectBestType. If no valid type is found, an
        // invalid TypeImpl is returned.
        static const TypeImpl& GetBestOperandType(Operation p_operator, 
                                                  const TypeImpl& p_operandType);

        // Return the type of a binary operator result given an operator and
        // an operand type. If the operand type is not a valid operand type for
        // the operator, the return type is undefined.
        static const TypeImpl& GetResultType(Operation p_operator,
                                             const TypeImpl& p_operandType);
    };
}

#endif

