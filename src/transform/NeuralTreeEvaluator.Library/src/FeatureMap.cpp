#include "FeatureMap.h"
#include "MinimalFeatureMap.h"

UInt32 FeatureMapGetFeatureCount(MinimalFeatureMap* featureMap)
{
    if (featureMap)
    {
        return featureMap->GetNumberOfFeatures();
    }
    return 0;
}

UInt32 FeatureMapGetFeatureNameMaxLength(MinimalFeatureMap* featureMap)
{
    if (featureMap)
    {
        UInt32 max = 0;
        UInt32 num = featureMap->GetNumberOfFeatures();
        for (UInt32 i = 0; i<num; ++i)
        {
            UInt32 len = (UInt32)featureMap->GetFeatureName(i).length();
            if (len > max)
            {
                max = len;
            }
        }
        return max;
    }
    return 0;
}

bool FeatureMapGetFeatureIndex(
        MinimalFeatureMap* featureMap, 
        const char *featureName, 
        UInt32 *featureIndex)
{
    if (featureMap)
    {
        return featureMap->GetExistingFeatureIndex(featureName, *featureIndex);
    }
    return false;
}

bool FeatureMapGetFeatureName(
        MinimalFeatureMap* featureMap, 
        UInt32 featureIndex, 
        char *featureNameBuffer, 
        UInt32 sizeOfBuffer,
        UInt32* featureNameLength)
{
    memset(featureNameBuffer, 0x00, sizeOfBuffer);
    if (featureMap)
    {
        std::string name = featureMap->GetFeatureName(featureIndex);
        *featureNameLength = (UInt32)name.length();
        if (name.length() > sizeOfBuffer)
        {
            return false;
        }
        memcpy(featureNameBuffer, name.c_str(), name.length());
        return true;
    }
    return false;
}