#pragma once
#include "IFeatureMap.h"

namespace DynamicRank
{

// Interface to apply custom process on all features input. This is especially useful for 
// on-demand feature extraction.
class INeuralNetFeatures
{
public:
    // Process a given feature index. The feature index is the value returned from a FeatureMap.
    virtual void ProcessFeature(UInt32 p_featureIndex) = 0;

    // Same as above but will also pass in a vector of strings for all the segments.  
    // An empty vector means that the feature is part of the main L2 ranker.
    virtual void ProcessFeature(UInt32 p_featureIndex, const std::vector<std::string>& p_segments) = 0;
};

}

