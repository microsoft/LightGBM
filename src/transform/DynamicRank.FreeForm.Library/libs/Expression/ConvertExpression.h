#pragma once

#ifndef FREEFORM2_CONVERTEXPRESSION_H
#define FREEFORM2_CONVERTEXPRESSION_H

#include "Expression.h"

namespace FreeForm2
{
    // Base class to represent a type conversion.
    class ConversionExpression : public Expression
    {
    public:
        // Create a conversion expression, converting the expression given 
        // expression. The derived class must specify the conversion type with 
        // the GetType method.
        ConversionExpression(const Annotations& p_annotations,
                             const Expression& p_child);
        virtual ~ConversionExpression();

        virtual size_t GetNumChildren() const override;

        const TypeImpl& GetChildType() const;

        const Expression& GetChild() const;

    protected:
        // Accept a visitor for a derived class.
        template <typename Derived>
        void AcceptDerived(Visitor& p_visitor) const;

    private:
        // Child expression to convert.
        const Expression& m_child;
    };

    // Class to convert to a float.
    class ConvertToFloatExpression : public ConversionExpression
    {
    public:
        // Create a conversion expression, taking the expression to convert
        // to float.
        ConvertToFloatExpression(const Annotations& p_annotations,
                                 const Expression& p_child);

        virtual const TypeImpl& GetType() const override;

        virtual void Accept(Visitor& p_visitor) const override;
    };

    // Class to convert to an integer.
    class ConvertToIntExpression : public ConversionExpression
    {
    public:
        // Create a conversion expression, taking the expression to convert 
        // to int. Any floating point data is truncated.
        ConvertToIntExpression(const Annotations& p_annotations,
                               const Expression& p_child);

        virtual const TypeImpl& GetType() const override;

        virtual void Accept(Visitor& p_visitor) const override;
    };

    // Class to convert to an uint64.
    class ConvertToUInt64Expression : public ConversionExpression
    {
    public:
        // Create a conversion expression, taking the expression to convert 
        // to uint64. Any floating point data is truncated.
        ConvertToUInt64Expression(const Annotations& p_annotations,
                                  const Expression& p_child);

        virtual const TypeImpl& GetType() const override;

        virtual void Accept(Visitor& p_visitor) const override;
    };

    // Class to convert to an int32.
    class ConvertToInt32Expression : public ConversionExpression
    {
    public:
        // Create a conversion expression, taking the expression to convert 
        // to int32. Any data is truncated.
        ConvertToInt32Expression(const Annotations& p_annotations,
                                 const Expression& p_child);

        virtual const TypeImpl& GetType() const override;

        virtual void Accept(Visitor& p_visitor) const override;
    };

    // Class to convert to an uint32.
    class ConvertToUInt32Expression : public ConversionExpression
    {
    public:
        // Create a conversion expression, taking the expression to convert 
        // to int32. Any data is truncated.
        ConvertToUInt32Expression(const Annotations& p_annotations,
                                  const Expression& p_child);

        virtual const TypeImpl& GetType() const override;

        virtual void Accept(Visitor& p_visitor) const override;
    };

    // Class to convert to a boolean.
    class ConvertToBoolExpression : public ConversionExpression
    {
    public:
        // Create a conversion expression, taking the expression to convert 
        // to bool.
        ConvertToBoolExpression(const Annotations& p_annotations,
                                const Expression& p_child);

        virtual const TypeImpl& GetType() const override;

        virtual void Accept(Visitor& p_visitor) const override;
    };

    // Class to issue code for a given expression, then discard the resulting
    // value and return void.
    class ConvertToImperativeExpression : public ConversionExpression
    {
    public:
        // Create a conversion expression, taking the expression to convert 
        // to imperative.
        ConvertToImperativeExpression(const Annotations& p_annotations,
                                      const Expression& p_child);

        virtual const TypeImpl& GetType() const override;

        virtual void Accept(Visitor& p_visitor) const override;
    };
}

#endif
