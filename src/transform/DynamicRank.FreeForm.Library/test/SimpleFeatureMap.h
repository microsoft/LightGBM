#pragma once

#include "IFeatureMap.h"
#include "basic_types.h"

#include <map>
#include <vector>
#include <string>

#define _snprintf_s(a,b,c,...) snprintf(a,b,__VA_ARGS__)


class SimpleFeatureMap : public DynamicRank::IFeatureMap
{
private:
    std::map<std::string, UInt32> m_featureMap;
    std::vector<std::string> m_reverseFeatureMap;
    UInt32 m_numberOfFeatures;

public:
    SimpleFeatureMap();
    ~SimpleFeatureMap();

    // Get the index of a feature, or return false if there is no such feature.
    bool GetExistingFeatureIndex(const char* featureName, UInt32& featureIndex) const;

    // Convert the FeatureName to the FeatureIndex.
    bool ObtainFeatureIndex(const char* featureName, UInt32& featureIndex);

    // Convert the FeatureName to the FeatureIndex.
    bool ObtainFeatureIndex(const SIZED_STRING& featureName, UInt32& featureIndex);
    
    // Convert the FeatureIndex to the FeatureName.
    bool GetFeatureName(UInt32 featureIndex, char* featureName, UInt32 maxNameLength) const;
    const std::string& GetFeatureName(UInt32 featureIndex) const;

    // The number of features within the map.
    UInt32 GetNumberOfFeatures() const;
};
