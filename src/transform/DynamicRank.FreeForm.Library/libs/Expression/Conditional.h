#pragma once

#ifndef FREEFORM2_CONDITIONAL_H
#define FREEFORM2_CONDITIONAL_H

#include "Expression.h"

namespace FreeForm2
{
    class ConditionalExpression : public Expression
    {
    public:
        ConditionalExpression(const Annotations& p_annotations,
                              const Expression& p_condition,
                              const Expression& p_then,
                              const Expression& p_else);

        virtual void Accept(Visitor& p_visitor) const override;
        virtual size_t GetNumChildren() const override;
        virtual const TypeImpl& GetType() const override;

        // Gets the condition expression for this conditional.
        const Expression& GetCondition() const;

        // Gets the then expression for this conditional.
        const Expression& GetThen() const;

        // Gets the else expression for this conditional.
        const Expression& GetElse() const;

    private:
        // Condition used to choose between then/else.
        const Expression& m_condition;

        // Value if condition is true.
        const Expression& m_then;

        // Value if condition is false.
        const Expression& m_else;
    };    
};

#endif
