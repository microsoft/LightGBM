#pragma once

#ifndef FREEFORM2_OPERATOR_EXPRESSION_FACTORY_H
#define FREEFORM2_OPERATOR_EXPRESSION_FACTORY_H

#include "ConvertExpression.h"
#include "OperatorExpression.h"
#include "ExpressionFactory.h"
#include "FreeForm2Assert.h"
#include "SimpleExpressionOwner.h"

namespace FreeForm2
{
    // An OperatorExpressionFactory creates operator expressions over arguments.
    class OperatorExpressionFactory : public ExpressionFactory
    {
    public:
        // Constructor, taking an optional unary operator, and an optional
        // binary operator (though it is not sensible to provide neither).
        // p_multiArity indicates whether this expression factory allows
        // arbitrary numbers of parameters, which are combined with multiple
        // application of the binary operator.
        OperatorExpressionFactory(UnaryOperator::Operation p_unaryOp, 
                                  BinaryOperator::Operation p_binaryOp, 
                                  bool p_multiArity)
            : m_unaryOp(p_unaryOp), m_binaryOp(p_binaryOp), m_multiArity(p_multiArity)
        {
        }


    private:
        virtual 
        const Expression& 
        CreateExpression(const ProgramParseState::ExpressionParseState& p_state, 
                         SimpleExpressionOwner& p_owner,
                         TypeManager& p_typeManager) const override
        {
            if (p_state.m_children.size() == 1)
            {
                // Handle unary expressions.
                FF2_ASSERT(m_unaryOp != UnaryOperator::invalid);
                boost::shared_ptr<Expression> expr(
                        new UnaryOperatorExpression(Annotations(SourceLocation(1, p_state.m_offset)),
                                                    *p_state.m_children[0],
                                                    m_unaryOp));
                p_owner.AddExpression(expr);
                return *expr;
            }
            else
            {
                // Handle n-ary expressions
                FF2_ASSERT(m_binaryOp != BinaryOperator::invalid);
                boost::shared_ptr<Expression> expr 
                    = BinaryOperatorExpression::Alloc(Annotations(SourceLocation(1, p_state.m_offset)),
                                                      p_state.m_children,
                                                      m_binaryOp,
                                                      p_typeManager);
                p_owner.AddExpression(expr);
                return *expr;
            }
        }


        virtual std::pair<unsigned int, unsigned int> 
        Arity() const override
        {
            unsigned int upper = (m_binaryOp != BinaryOperator::invalid) 
                ? (m_multiArity ? UINT_MAX : 2) : 1;
            unsigned int lower = (m_unaryOp != UnaryOperator::invalid) ? 1 : 2;
            return std::make_pair(lower, upper);
        }


        // Unary operator used by created expressions to compile arithmetic.
        const UnaryOperator::Operation m_unaryOp;

        // Binary operator used by created expressions to compile arithmetic.
        const BinaryOperator::Operation m_binaryOp;

        // Whether this expression combines more than two arguments using
        // multiple application of the binary operator.
        bool m_multiArity;
    };
}

#endif
