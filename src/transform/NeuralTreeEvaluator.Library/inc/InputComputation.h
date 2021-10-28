/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#pragma once

#include <NeuralInput.h>

void InputGetFeatures(const DynamicRank::NeuralInput *input, UInt32 *features,
                      UInt32 sizeOfFeatures, UInt32 *featureCount);
bool InputIsCopy(const DynamicRank::NeuralInput *input);
double InputEvaluate(const DynamicRank::NeuralInput *input,
                     UInt32 *featureValues);
void InputEvaluateMany(const DynamicRank::NeuralInput *input,
                       UInt32 **featureValues, double *outputs, UInt32 count);