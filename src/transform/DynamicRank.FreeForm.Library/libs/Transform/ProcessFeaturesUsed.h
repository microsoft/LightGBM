#pragma once

#ifndef FREEFORM2_PROCESS_FEATURES_USED_H
#define FREEFORM2_PROCESS_FEATURES_USED_H

#include "NoOpVisitor.h"

namespace DynamicRank
{
    class IFeatureMap;
    class INeuralNetFeatures;
};

namespace FreeForm2
{
    // Class to collect the set of features used by a program.
    class ProcessFeaturesUsedVisitor : public NoOpVisitor
    {
    public:
        ProcessFeaturesUsedVisitor(DynamicRank::INeuralNetFeatures& p_features);
 
        // Methods inherited from Visitor.
        virtual void Visit(const FeatureRefExpression& p_expr);

    private:
        DynamicRank::INeuralNetFeatures& m_features;
    };
}

#endif
