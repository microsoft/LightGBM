#pragma once

#ifndef FREEFORM2_DECLARATION_H
#define FREEFORM2_DECLARATION_H

#include "Expression.h"

namespace FreeForm2
{
    class DeclarationExpression : public Expression
    {
    public:
        // Create a declaration expression from a type (which may be
        // Type::Unknown) and an initialiser.  p_voidValue controls whether 
        // the DeclarationExpression evaluates to a void value (imperatively), 
        // or to the p_init expression (functionally).
        DeclarationExpression(const Annotations& p_annotations,
                              const TypeImpl& p_type, 
                              const Expression& p_init,
                              bool p_voidValue,
                              VariableID p_id,
                              size_t p_version);

        // Virtual methods inherited from Expression.
        virtual void Accept(Visitor& p_visitor) const override;
        virtual size_t GetNumChildren() const override;
        virtual const TypeImpl& GetType() const override;

        // Whether this expression evaluates to void, or the value m_init;
        bool HasVoidValue() const;

        // Return the type of the variable declared (not the type of this 
        // declaration expression, which is significantly different).
        const TypeImpl& GetDeclaredType() const;

        // Return the initialization expression.
        const Expression& GetInit() const;

        // Gets this declaration's unique identifier and value version.
        VariableID GetId() const;
        size_t GetVersion() const;

    private:

        // Type of the variable declared by this expression.  (Note: this is
        // *not* the type of the declaration expression);
        const TypeImpl& m_declType;

        // Initialisation expression.
        const Expression& m_init;

        // Whether this expression evaluates to void, or the value m_init;
        bool m_voidValue;

        // A unique identificator to allow separation of allocation and usage.
        const VariableID m_id;

        // A unique version number associated with a particular
        // value for this variable.
        const size_t m_version;
    };
};

#endif
