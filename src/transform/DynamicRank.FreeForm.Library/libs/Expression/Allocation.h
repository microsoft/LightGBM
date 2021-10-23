#pragma once
#ifndef FREEFORM2_ARRAYALLOCATIONWRAPPER_H
#define FREEFORM2_ARRAYALLOCATIONWRAPPER_H

#include "Expression.h"
#include "Visitor.h"
#include <vector>

namespace FreeForm2
{
    class AllocationExpression;

    // Wraps the original expression and decouples array allocation from
    // initialization.
    class Allocation : public boost::noncopyable
    {
    public:
        enum AllocationType
        {
            ArrayLiteral,
            FeatureArray,
            ExternArray,
            LiteralStream,
            LiteralWord,
            Declaration
        };

        Allocation(AllocationType p_allocType,
                   VariableID p_id,
                   const TypeImpl& p_type);

        Allocation(AllocationType p_allocType,
                   VariableID p_id,
                   const TypeImpl& p_type,
                   size_t p_children);

        // Gets the type of the element to be allocated.
        const TypeImpl& GetType() const;

        // Gets the number of children of the allocation.
        size_t GetNumChildren() const;

        // Gets the type of the allocation.
        AllocationType GetAllocationType() const;
        
        // Gets the identifier of the array.
        VariableID GetAllocationId() const;

    private:
        // The type of the allocation.
        const AllocationType m_allocType;

        // The number of children.
        size_t m_children;

        // The array identificator.
        const VariableID m_id;

        // The type of the element to be allocated.
        const TypeImpl& m_type;
    };
}

#endif
