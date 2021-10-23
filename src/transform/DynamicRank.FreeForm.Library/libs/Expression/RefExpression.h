#pragma once

#ifndef FREEFORM2_REFEXPRESSION_H
#define FREEFORM2_REFEXPRESSION_H

#include "Expression.h"

namespace FreeForm2
{
    // Class representing a reference to a feature.
    class FeatureRefExpression : public Expression
    {
    public:
        FeatureRefExpression(const Annotations& p_annotations,
                             UInt32 p_index);

        virtual const TypeImpl& GetType() const override;

        virtual size_t GetNumChildren() const override;

        virtual void Accept(Visitor& p_visitor) const override;

        UInt32 m_index;
    };

    // Class representing a reference to a stack location. We keep track of only 
    // a stack slot, so that the value of the expression can be generated 
    // during compilation, rather than parsing, and looked up using the stack 
    // slot as identifier.
    class VariableRefExpression : public Expression
    {
    public:
        // Construct a stack expression from a stack slot.
        VariableRefExpression(const Annotations& p_annotations,
                              VariableID p_id,
                              size_t p_version,
                              const TypeImpl& p_type);
        virtual ~VariableRefExpression();

        // Methods inherited from Expression (note that VariableRefExpression 
        // can generate references).
        virtual size_t GetNumChildren() const override;
        virtual void Accept(Visitor& p_visitor) const override;
        virtual void AcceptReference(Visitor&) const override;
        virtual const TypeImpl& GetType() const override;

        VariableID GetId() const;
        size_t GetVersion() const;

    private:
        // ID assigned to the value.
        VariableID m_id;

        // A unique version number associated with a particular
        // value for this variable.
        const size_t m_version;

        // Type of the value.
        const TypeImpl& m_type;
    };

    // This class represents an expression which refers to the not-yet-
    // instantiated object for the scope being compiled (for instance, a state
    // machine will use a ThisExpression to refer to a member inside the state
    // machine, as the machine is not instantiated at this point).
    class ThisExpression : public Expression
    {
    public:
        // Construct a ThisExpression from a compound type.
        ThisExpression(const Annotations& p_annotations,
                       const TypeImpl& p_type);

        // Methods inherited from Visitor.
        virtual size_t GetNumChildren() const override;
        virtual void Accept(Visitor& p_visitor) const override;
        virtual void AcceptReference(Visitor&) const override;
        virtual const TypeImpl& GetType() const override;

    private:
        // The type of the object.
        const TypeImpl& m_type;
    };
}

#endif
