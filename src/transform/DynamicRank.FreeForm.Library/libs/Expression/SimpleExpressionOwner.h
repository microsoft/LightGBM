#pragma once

#ifndef FREEFORM2_SIMPLEEXPRESSIONOWNER
#define FREEFORM2_SIMPLEEXPRESSIONOWNER

#include "Expression.h"
#include <vector>

namespace FreeForm2
{
    // Straight-forward expression owner, that keeps shared pointers
    // to given expressions.
    class SimpleExpressionOwner : public ExpressionOwner
    {
    public:
        // Transfer ownership of the given expression to the expression owner.
        const Expression* AddExpression(const boost::shared_ptr<const Expression>& p_expr)
        {
            m_exp.push_back(p_expr);
            return m_exp.back().get();
        }

    private:
        // Vector of managed expressions.
        std::vector<boost::shared_ptr<const Expression>> m_exp;
    };
}

#endif
