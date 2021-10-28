/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#pragma once

#include "basic_types.h"

namespace DynamicRank {

// A Feature is defined by any of the following three parameters
// FeatureName : The Name of the Feature
// FeatureIndex : The Index of the Feature according to the FeatureMap
// FeatureName <-> FeatureIndex conversion is a combination of the above two
class IFeatureMap {
 public:
  // Destructor
  virtual ~IFeatureMap() {}

  // Convert the FeatureName to the FeatureIndex.
  virtual bool ObtainFeatureIndex(const char *p_featureName,
                                  UInt32 &p_featureIndex) = 0;

  // Convert the FeatureName to the FeatureIndex.
  virtual bool ObtainFeatureIndex(const SIZED_STRING &p_featureName,
                                  UInt32 &p_featureIndex) = 0;

  // Convert the FeatureIndex to the FeatureName.
  virtual bool GetFeatureName(UInt32 p_featureIndex, char *p_featureName,
                              UInt32 p_maxNameLength) const = 0;

  // The number of features within the map.
  virtual UInt32 GetNumberOfFeatures() const = 0;
};
}  // namespace DynamicRank
