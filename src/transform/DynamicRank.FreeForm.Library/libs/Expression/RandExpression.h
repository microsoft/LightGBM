#pragma once

#ifndef FREEFORM2_RAND_EXPRESSION_H
#define FREEFORM2_RAND_EXPRESSION_H

#include "Expression.h"

namespace FreeForm2
{
    // The RandFloatExpression generates a random float
    // in the range of 0 to 1 inclusive.
    class RandFloatExpression : public Expression
    {
    public:
        // Methods inherited from Expression.
        void Accept(Visitor& p_visitor) const override;
        virtual const TypeImpl& GetType() const override;
        virtual size_t GetNumChildren() const override;

        // Get a reference to the singleton for RandFloatExpression.
        static const RandFloatExpression& GetInstance();

    private:
        // Private constructor for singleton class.
        RandFloatExpression(const Annotations& p_annotations);
    };

    // The RandIntExpression generates a random integer
    // in the range specified inclusive of the lower bound
    // and upper bound exclusive.
    class RandIntExpression: public Expression
    {
    public:
        // Constructor for RandIntExpression.
        RandIntExpression(const Annotations& p_annotations,
                          const Expression& p_lowerBoundExpression,
                          const Expression& p_upperBoundExpression);

        // Methods inherited from Expression.
        void Accept(Visitor& p_visitor) const override;
        virtual const TypeImpl& GetType() const override;
        virtual size_t GetNumChildren() const override;
        
    private:
        
        // Lower bound expression to generate a random number.
        const Expression& m_lowerBoundExpression;

        // Upper bound expression to generate a random number.
        const Expression& m_upperBoundExpression;
    };
};

#endif
