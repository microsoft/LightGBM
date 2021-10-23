#include "ProcessFeaturesUsed.h"

#include "RefExpression.h"
#include <INeuralNetFeatures.h>

FreeForm2::ProcessFeaturesUsedVisitor::ProcessFeaturesUsedVisitor(
        DynamicRank::INeuralNetFeatures& p_features)
    : m_features(p_features)
{
}


void 
FreeForm2::ProcessFeaturesUsedVisitor::Visit(const FeatureRefExpression& p_expr)
{
    m_features.ProcessFeature(p_expr.m_index);
}
