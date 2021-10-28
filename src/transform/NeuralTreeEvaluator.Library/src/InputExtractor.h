/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#pragma once

#include <NeuralInput.h>
#include <vector>
#include "MinimalFeatureMap.h"

#include <string>
#include <map>
#include <vector>
#include <boost/foreach.hpp>
#include <boost/scoped_array.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/noncopyable.hpp>
#include <set>

using namespace DynamicRank;

class InputExtractor
{
private:
    // The configuration file from which we update these.
    DynamicRank::Config *m_config;
    // The list of inputs corresponding to evaluators.
    std::vector<boost::shared_ptr<const DynamicRank::NeuralInput> > m_inputs;
    // The feature map.
    boost::shared_ptr<MinimalFeatureMap> m_featureMap;
    InputExtractor(
        DynamicRank::Config *config,
        boost::shared_ptr<MinimalFeatureMap> featureMap,
        std::vector<boost::shared_ptr<const DynamicRank::NeuralInput> > &inputs);

public:
    ~InputExtractor(void);

    static InputExtractor *CreateFromConfig(DynamicRank::Config *config);
    static InputExtractor *CreateFromInputStr(const string &str);
    static InputExtractor *CreateFromFreeform2(const char *freeform);

    MinimalFeatureMap *GetFeatureMap() const;
    UInt32 GetInputCount() const;
    const DynamicRank::NeuralInput *GetInput(UInt32 index) const;
    const std::string GetInputName(UInt32 index) const;
};