/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include "ProcessFeaturesUsed.h"

#include <INeuralNetFeatures.h>

#include "RefExpression.h"

FreeForm2::ProcessFeaturesUsedVisitor::ProcessFeaturesUsedVisitor(
    DynamicRank::INeuralNetFeatures &p_features)
    : m_features(p_features) {}

void FreeForm2::ProcessFeaturesUsedVisitor::Visit(
    const FeatureRefExpression &p_expr) {
  m_features.ProcessFeature(p_expr.m_index);
}
