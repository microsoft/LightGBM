/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include "SimpleFeatureMap.h"

#include <algorithm>

SimpleFeatureMap::SimpleFeatureMap() : m_numberOfFeatures(0) {}

SimpleFeatureMap::~SimpleFeatureMap() {}

bool SimpleFeatureMap::GetExistingFeatureIndex(const char *featureName,
                                               UInt32 &featureIndex) const {
  std::string featureNameStr(featureName);
  std::map<std::string, UInt32>::const_iterator it =
      m_featureMap.find(featureNameStr);

  if (it == m_featureMap.end()) {
    return false;
  }
  featureIndex = it->second;
  return true;
}

bool SimpleFeatureMap::ObtainFeatureIndex(const char *featureName,
                                          UInt32 &featureIndex) {
  const std::string featureNameStr(featureName);
  std::map<std::string, UInt32>::iterator it =
      m_featureMap.find(featureNameStr);
  if (it != m_featureMap.end()) {
    featureIndex = it->second;
  } else {
    UInt32 iFeature = m_numberOfFeatures;
    std::pair<std::string, UInt32> pair(featureNameStr, iFeature);
    m_featureMap.insert(pair);
    m_reverseFeatureMap.push_back(std::string(featureName));
    m_numberOfFeatures++;
    featureIndex = iFeature;
  }
  return true;
}

bool SimpleFeatureMap::ObtainFeatureIndex(const SIZED_STRING &featureName,
                                          UInt32 &featureIndex) {
  std::string localFeatureName((const char *)featureName.pbData,
                               featureName.cbData);
  return ObtainFeatureIndex(localFeatureName.c_str(), featureIndex);
}

bool SimpleFeatureMap::GetFeatureName(UInt32 featureIndex, char *featureName,
                                      UInt32 maxNameLength) const {
  if (featureIndex < m_numberOfFeatures) {
    _snprintf_s(featureName, maxNameLength, _TRUNCATE, "%s",
                m_reverseFeatureMap[featureIndex].c_str());
    return true;
  }
  return false;
}

const std::string &SimpleFeatureMap::GetFeatureName(UInt32 featureIndex) const {
  return m_reverseFeatureMap[featureIndex];
}

UInt32 SimpleFeatureMap::GetNumberOfFeatures() const {
  return m_numberOfFeatures;
}
