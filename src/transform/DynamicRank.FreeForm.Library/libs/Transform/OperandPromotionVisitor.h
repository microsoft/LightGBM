#pragma once

#ifndef FREEFORM2_OPERAND_PROMOTION_VISITOR_H
#define FREEFORM2_OPERAND_PROMOTION_VISITOR_H

#include "CopyingVisitor.h"

#include <stack>
#include <boost/shared_ptr.hpp>

namespace FreeForm2
{
    class SimpleExpressionOwner;
    class ExpressionOwner;
    class Expression;

    class OperandPromotionVisitor : public CopyingVisitor
    {
    public:
        // Allow promotion in binary operator.
        virtual bool AlternativeVisit(const BinaryOperatorExpression& p_expr);
        virtual void Visit(const BinaryOperatorExpression& p_expr);
        
        // Allow promotion in the function call expression.
        virtual void Visit(const FunctionCallExpression& p_expr);
        
        // Allow promotion in conditional.
        virtual void Visit(const ConditionalExpression& p_expr);
        
        // Allow promotion in loop structures.
        virtual void Visit(const RangeReduceExpression& p_expr);
        virtual void Visit(const ForEachLoopExpression& p_expr);
        virtual void Visit(const ComplexRangeLoopExpression& p_expr);

        // Promote void type object method expressions to statements.
        // virtual void Visit(const ObjectMethodExpression& p_expr);
    };
}

#endif
