#pragma once

#ifndef FREEFORM2_UNARY_OPERATOR_H
#define FREEFORM2_UNARY_OPERATOR_H

namespace FreeForm2
{
    class TypeImpl;

    class UnaryOperator
    {
    public:
        enum Operation
        {
            minus,
            log,
            log1,
            abs,
            round,
            trunc,
            _not,
            bitnot,
            tanh,

            invalid
        };

        // Select the best operand type for an operator. Best is defined in 
        // terms of TypeUtil::SelectBestType. If no valid type is found, an
        // invalid TypeImpl is returned.
        static const TypeImpl& GetBestOperandType(Operation p_operator, const TypeImpl& p_operand);

        // Return the type of a unary operator result given an operator and
        // an operand type. If the operand type is not a valid operand type for
        // the operator, the return type is undefined.
        static const TypeImpl& GetReturnType(Operation p_operator, const TypeImpl& p_operand);

    };
}

#endif

