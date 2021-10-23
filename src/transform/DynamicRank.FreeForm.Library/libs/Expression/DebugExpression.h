#pragma once

#ifndef FREEFORM2_DEBUG_EXPRESSION_H
#define FREEFORM2_DEBUG_EXPRESSION_H

#include "Expression.h" 
#include <string>

namespace FreeForm2
{
    // A DebugExpression assists developers by allowing expressions to be
    // debugged. The exact method of debugging depends on other compiler 
    // settings, but generally debug instrumentation should provide the
    // original text of the expression being debugged, and the value of that
    // expression.
    class DebugExpression : public Expression
    {
    public:
        // Construct a DebugExpression which will provide debugging 
        // capabilities for an expression. The child text refers to the 
        // original text of the expression to debug.
        DebugExpression(const Annotations& p_annotations,
                        const Expression& p_child,
                        const std::string& p_childText);

        // Methods inherited from Expression.
        virtual void Accept(Visitor&) const override;
        virtual const TypeImpl& GetType() const override;
        virtual size_t GetNumChildren() const override;

        // Get the child expression.
        const Expression& GetChild() const;

        // Get the text of the child expression to be printed for debugging
        // purposes.
        const std::string& GetChildText() const;

    private:
        // The child expression.
        const Expression& m_child;

        // The original text of the child expression.
        std::string m_childText;
    };
}

#endif
