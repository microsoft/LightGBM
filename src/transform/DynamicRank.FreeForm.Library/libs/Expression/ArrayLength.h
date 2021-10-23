#pragma once

#ifndef FREEFORM2_ARRAY_LENGTH_H
#define FREEFORM2_ARRAY_LENGTH_H

#include "Expression.h"

namespace FreeForm2
{
    class ArrayLengthExpression : public Expression
    {
    public:
        // Construct an array length expression given the array in question.
        ArrayLengthExpression(const Annotations& p_annotations,
                              const Expression& p_array);

        virtual void Accept(Visitor& p_visitor) const override;

        // Return the type of an array-length expression (int).
        virtual const TypeImpl& GetType() const override;

        // Return the number of child nodes for this expression.
        virtual size_t GetNumChildren() const override;

        // Get the array
        const Expression& GetArray() const 
        { 
            return m_array; 
        }

    private:
        // Array that we're calculating the length of.
        const Expression& m_array;
    };
};

#endif

