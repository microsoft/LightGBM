/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#pragma once
#include "InputExtraction.h"
#include "InputComputation.h"
#include <assert.h>
#include <LightGBM/utils/log.h>
#include <LightGBM/utils/common.h>
using namespace LightGBM;

void EvaluateBasicInputs(string fea_spec_path)
{
    DynamicRank::Config *config = DynamicRank::Config::GetRawConfiguration(Common::LoadStringFromFile(fea_spec_path.c_str()));
    assert(config->DoesSectionExist("Input:3"));
    InputExtractor *extractor = InputExtractorCreateFromInputStr(Common::LoadStringFromFile(fea_spec_path.c_str()));
    assert(extractor != NULL);
}

void EvaluateFreeFormHelper(InputExtractor *extractor)
{
    MinimalFeatureMap *featureMap = InputExtractorGetFeatureMap(extractor);
    int inputCount = InputExtractorGetInputCount(extractor);
    Log::Info("Input count:  %s", to_string(inputCount).c_str());
    assert(inputCount == 1);

    const DynamicRank::NeuralInput *input = InputExtractorGetInput(extractor, 0);
    UInt32 features[2];
    features[0] = 0;
    features[1] = 0;
    double value = InputEvaluate(input, features);
    Log::Info("Evaluated value: %s", to_string(value).c_str());
    assert(value == 1);

    features[0] = 1;
    value = InputEvaluate(input, features);
    Log::Info("Evaluated value: %s", to_string(value).c_str());
    assert(value == 0);
}

void EvaluateStandaloneFreeFormV2()
{
    InputExtractor *extractor = InputExtractorCreateFromFreeformV2("(if (== Foo Bar) 1 0)");
    EvaluateFreeFormHelper(extractor);
}

int main()
{
    EvaluateBasicInputs("./data/TrainInputIni");
    EvaluateStandaloneFreeFormV2();
    return 0;
}
