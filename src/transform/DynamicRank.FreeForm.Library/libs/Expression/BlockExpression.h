#pragma once

#ifndef FREEFORM2_BLOCK_EXPRESSION_H
#define FREEFORM2_BLOCK_EXPRESSION_H

#include <boost/shared_ptr.hpp>
#include "Expression.h"

namespace FreeForm2
{
    // A block expression is a series of expressions.
    class BlockExpression : public Expression
    {
    public:
        // Allocate a block expression for the given array of child expressions,
        // with p_numBound indicating the number of symbols bound by (and not
        // scoped within) the immediate children of the block expression.
        static boost::shared_ptr<BlockExpression> 
        Alloc(const Annotations& p_annotations,
              const Expression** p_children, 
              unsigned int p_numChildren, 
              unsigned int p_numBound);

        // Return the number of symbols bound by immediate children of this 
        // block expression, and left open.
        unsigned int GetNumBound() const;

        // Methods inherited from Expression.
        virtual size_t GetNumChildren() const override;
        virtual void Accept(Visitor& p_visitor) const override;
        virtual const TypeImpl& GetType() const override;

        // Gets the p_index-th child of this expression.
        const Expression& GetChild(unsigned int p_index) const;

    private:
        // Private ctor: use Alloc to create block expressions.
        BlockExpression(const Annotations& p_annotations,
                        const Expression** p_children,
                        unsigned int p_numChildren, 
                        unsigned int p_numBound);

        // Custom deallocation method.
        static void DeleteAlloc(BlockExpression* p_allocated);
 
        // The return type of the block.
        const TypeImpl* m_returnType;

        // Number of children of this block.
        unsigned int m_numChildren;
 
        // Number of symbols left bound by children of the block expression.
        unsigned int m_numBound;

        // Children, allocated via struct hack.
        const Expression* m_children[1];
    };
};

#endif

