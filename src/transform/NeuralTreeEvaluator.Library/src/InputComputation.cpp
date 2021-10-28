/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include "InputComputation.h"
#include <NeuralInput.h>

void InputGetFeatures(const DynamicRank::NeuralInput *input, UInt32 *features, UInt32 sizeOfFeatures, UInt32 *featureCount)
{
    if (input == NULL)
    {
        *featureCount = 0;
        return;
    }
    std::vector<UInt32> featureList;
    input->GetAllAssociatedFeatures(featureList);
    *featureCount = (UInt32)featureList.size();
    for (UInt32 i = 0; i < *featureCount && i < sizeOfFeatures; ++i)
    {
        features[i] = featureList[i];
    }
}

bool InputIsCopy(const DynamicRank::NeuralInput *input)
{
    if (input == NULL)
    {
        return false;
    }
    const DynamicRank::NeuralInput *baseInput = input;
    const DynamicRank::NeuralInputCached *cachedInput = dynamic_cast<const DynamicRank::NeuralInputCached *>(input);
    if (cachedInput != NULL)
    {
        baseInput = cachedInput->GetBaseInput();
    }

    const DynamicRank::NeuralInputLinear *linearInput = dynamic_cast<const DynamicRank::NeuralInputLinear *>(baseInput);

    // Do a quick check for the identity function once we establish that this is a linear input.
    return linearInput != NULL && linearInput->EvaluateInput(0) == 0.0 && linearInput->EvaluateInput(1) == 1.0;
}

double InputEvaluate(const DynamicRank::NeuralInput *input, UInt32 *featureValues)
{
    return input->Evaluate(featureValues);
}

void InputEvaluateMany(const DynamicRank::NeuralInput *input, UInt32 **featureValues, double *outputs, UInt32 count)
{
    for (UInt32 i = 0; i < count; ++i)
    {
        outputs[i] = input->Evaluate(featureValues[i]);
    }
}
