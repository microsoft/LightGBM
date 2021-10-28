/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#pragma once

#include "basic_types.h"

class MinimalFeatureMap;

UInt32 FeatureMapGetFeatureCount(MinimalFeatureMap *featureMap);
UInt32 FeatureMapGetFeatureNameMaxLength(MinimalFeatureMap *featureMap);
bool FeatureMapGetFeatureIndex(
    MinimalFeatureMap *featureMap,
    const char *featureName,
    UInt32 *featureIndex);
bool FeatureMapGetFeatureName(
    MinimalFeatureMap *featureMap,
    UInt32 featureIndex,
    char *featureNameBuffer,
    UInt32 sizeOfBuffer,
    UInt32 *featureNameLength);