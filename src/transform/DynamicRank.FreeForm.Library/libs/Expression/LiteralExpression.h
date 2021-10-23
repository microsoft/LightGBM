#pragma once

#ifndef FREEFORM2_LITERALEXPRESSION_H
#define FREEFORM2_LITERALEXPRESSION_H

#include "Expression.h"
#include "FreeForm2Result.h"

namespace FreeForm2
{
    // This class represents a literal expression of any leaf type.
    class LeafTypeLiteralExpression : public Expression
    {
    public:
        explicit LeafTypeLiteralExpression(const Annotations& p_annotations, Result::IntType p_value);
        explicit LeafTypeLiteralExpression(const Annotations& p_annotations, Result::UInt64Type p_value);
        explicit LeafTypeLiteralExpression(const Annotations& p_annotations, Result::Int32Type p_value);
        explicit LeafTypeLiteralExpression(const Annotations& p_annotations, Result::UInt32Type p_value);
        explicit LeafTypeLiteralExpression(const Annotations& p_annotations, Result::FloatType p_value);
        explicit LeafTypeLiteralExpression(const Annotations& p_annotations, Result::BoolType p_value);

        virtual bool IsConstant() const override;
        virtual ConstantValue GetConstantValue() const override;

    private:
        ConstantValue m_value;
    };

    class LiteralIntExpression : public LeafTypeLiteralExpression
    {
    public:
        // Construct a literal expression from an int.
        explicit LiteralIntExpression(const Annotations& p_annotations, Result::IntType p_val);

        virtual const TypeImpl& GetType() const override;

        virtual size_t GetNumChildren() const override;

        virtual void Accept(Visitor& p_visitor) const override;
    };

    class LiteralUInt64Expression : public LeafTypeLiteralExpression
    {
    public:
        // Construct a literal expression from an uint64.
        explicit LiteralUInt64Expression(const Annotations& p_annotations, Result::UInt64Type p_val);

        virtual const TypeImpl& GetType() const override;

        virtual size_t GetNumChildren() const override;

        virtual void Accept(Visitor& p_visitor) const override;
    };

    class LiteralInt32Expression : public LeafTypeLiteralExpression
    {
    public:
        // Construct a literal expression from an int32.
        explicit LiteralInt32Expression(const Annotations& p_annotations, Result::Int32Type p_val);

        virtual const TypeImpl& GetType() const override;

        virtual size_t GetNumChildren() const override;

        virtual void Accept(Visitor& p_visitor) const override;
    };

    class LiteralUInt32Expression : public LeafTypeLiteralExpression
    {
    public:
        // Construct a literal expression from an uint32.
        explicit LiteralUInt32Expression(const Annotations& p_annotations, Result::UInt32Type p_val);

        virtual const TypeImpl& GetType() const override;

        virtual size_t GetNumChildren() const override;

        virtual void Accept(Visitor& p_visitor) const override;
    };

    class LiteralFloatExpression : public LeafTypeLiteralExpression
    {
    public:
        // Construct a literal expression from a float.
        explicit LiteralFloatExpression(const Annotations& p_annotations, Result::FloatType p_val);

        virtual const TypeImpl& GetType() const override;

        virtual size_t GetNumChildren() const override;

        virtual void Accept(Visitor& p_visitor) const override;
    };

    class LiteralBoolExpression : public LeafTypeLiteralExpression
    {
    public:
        // Construct a literal expression from an int.
        explicit LiteralBoolExpression(const Annotations& p_annotations, Result::BoolType p_val);

        virtual const TypeImpl& GetType() const override;

        virtual size_t GetNumChildren() const override;

        virtual void Accept(Visitor& p_visitor) const override;
    };

    class LiteralVoidExpression : public Expression
    {
    public:
        // Get the single instance of LiteralVoidExpression.  There's no point
        // creating many instances of this expression, since they're all the
        // same.
        static const LiteralVoidExpression& GetInstance();

        // Virtual methods inherited from Expression.
        virtual const TypeImpl& GetType() const override;
        virtual size_t GetNumChildren() const override;
        virtual void Accept(Visitor& p_visitor) const override;

    private:
        LiteralVoidExpression(const Annotations& p_annotations);
    };

    class LiteralWordExpression : public Expression
    {
    public:
        // Construct a literal expression from a set of integers.
        LiteralWordExpression(const Annotations& p_annotations,
                              const Expression& p_word, 
                              const Expression& p_offset,
                              const Expression* p_attribute,
                              const Expression* p_length, 
                              const Expression* p_candidate,
                              VariableID p_variableId);

        // Construct a literal instance header from integers.
        LiteralWordExpression(const Annotations& p_annotations,
                              const Expression& p_instanceHeaderLength, 
                              const Expression& p_instanceHeaderOffset,
                              VariableID p_variableId);

        virtual const TypeImpl& GetType() const override;
        virtual size_t GetNumChildren() const override;
        virtual void Accept(Visitor& p_visitor) const override;

        // Gets the integer identificator for this literal.
        VariableID GetId() const;

        // Whether this literal represents a stream instance header or an 
        // ordinary word occurrence.
        bool m_isHeader;

        // Members in a WordOccurrence.  TODO: ensure that a WordOccurrence
        // struct is of the same size.  Note that we've overloaded word to carry
        // instance header lengths, and offset to carry instance header counts.
        const Expression& m_word;
        const Expression& m_offset;
        const Expression* m_attribute;
        const Expression* m_length;
        const Expression* m_candidate;

    private:
        // The integer identificator for this literal.
        VariableID m_variableID;
    };

    class LiteralStreamExpression : public Expression
    {
    public:
        virtual const TypeImpl& GetType() const override;
        virtual size_t GetNumChildren() const override;
        virtual void Accept(Visitor& p_visitor) const override;

        static boost::shared_ptr<LiteralStreamExpression> 
        Alloc(const Annotations& p_annotations, const Expression** p_children, size_t p_numChildren, VariableID p_id);

        const Expression* const* GetChildren() const;

        VariableID GetId() const;

    private:
        LiteralStreamExpression(const Annotations& p_annotations,
                                const Expression** p_children, 
                                size_t p_numChildren,
                                VariableID p_id);

        static void DeleteAlloc(LiteralStreamExpression* p_allocated);
 
        // Number of children of this node.
        size_t m_numChildren;

        // A unique identificator to allow separation of allocation and usage.
        const VariableID m_id;

        // Array of children of this node, allocated using struct hack.
        const Expression* m_children[1];
    };

    // Represents an instance header in a stream.
    class LiteralInstanceHeaderExpression : public Expression
    {
    public:
        // Constructor.
        LiteralInstanceHeaderExpression(const Annotations& p_annotations,
                                        const Expression& p_instanceCount,
                                        const Expression& p_rank,
                                        const Expression& p_instanceLength);

        // Methods inherited from Expression class.
        virtual const TypeImpl& GetType() const override;
        virtual size_t GetNumChildren() const override;
        virtual void Accept(Visitor& p_visitor) const override;

        // Properties associated with instance headers.
        const Expression& m_instanceCount;
        const Expression& m_rank;
        const Expression& m_instanceLength;
    };
}

#endif 
