#pragma once

#ifndef FREEFORM2_PROGRAM_H
#define FREEFORM2_PROGRAM_H

#include "AllocationVisitor.h"
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include "FreeForm2Type.h"
#include <vector>

namespace DynamicRank
{
    class IFeatureMap;
    class INeuralNetFeatures;
}

namespace FreeForm2
{
    class Allocation;
    class Expression;
    class ExpressionOwner;
    class TypeManager;

    // A ProgramImpl is essentially just the concrete instantiation of the
    // Program class, into which we parse expressions.
    class ProgramImpl : boost::noncopyable
    {
    public:
        ProgramImpl(const Expression& p_exp, 
                    boost::shared_ptr<ExpressionOwner> p_owner,
                    boost::shared_ptr<TypeManager> p_typeManager,
                    DynamicRank::IFeatureMap& p_map);

        const Type& GetType() const;

        void 
        ProcessFeaturesUsed(DynamicRank::INeuralNetFeatures& p_features) const;

        const Expression& GetExpression() const;

        DynamicRank::IFeatureMap&
        GetFeatureMap() const;

        const std::vector<boost::shared_ptr<Allocation>>& GetAllocations() const;

    private:
        // Top-level type implementation of this program.
        const TypeImpl& m_typeImpl;

        // Top-level type of this program.
        Type m_type;

        // Pointer to root of parsed expression tree.
        const Expression* m_exp;
       
        // Expression owner.
        boost::shared_ptr<ExpressionOwner> m_owner;

        // Type manager.
        boost::shared_ptr<TypeManager> m_typeManager;

        // Feature map used to compile expression, used to print program
        // information if exceptions occur.
        DynamicRank::IFeatureMap& m_map;

        // A visitor to extract all the allocations from the program.
        const AllocationVisitor m_allocationVisitor;
    };
}

#endif